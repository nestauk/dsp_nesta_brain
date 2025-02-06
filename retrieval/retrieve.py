from __future__ import annotations

import asyncio
import importlib
import os
import re

from collections import OrderedDict
from typing import List
from typing import Optional
from typing import Union

import lancedb

from config import DB_PATH
from config import DEFAULT_EMBEDDINGS_MODEL
from config import PROJECT
from dotenv import load_dotenv
from dsp_nesta_brain import logger
from lancedb.db import LanceDBConnection
from lancedb.table import LanceTable
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import LanceDB
from langchain_core.messages import HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import MessagesState
from openai import OpenAI
from retrieval.db.schema.nesta_brain import Chunk as NestaBrainChunk
from retrieval.db.schema.nesta_brain import MissionProject
from retrieval.db.schema.policy_atlas import Activity
from utils import unique


if PROJECT == "NESTA_BRAIN":
    Chunk = NestaBrainChunk
    chunk_table_name = "chunk"
    default_merge = True
    SCHEMA_MODULE = importlib.import_module("retrieval.db.schema.nesta_brain")

elif PROJECT == "POLICY_ATLAS":
    Chunk = Activity
    chunk_table_name = "activity"
    default_merge = False  # activity records were not split into separate chunks,so no need to merge
    SCHEMA_MODULE = importlib.import_module("retrieval.db.schema.policy_atlas")

table_name_to_schema_class_map = SCHEMA_MODULE.table_name_to_schema_class_map


class RetrieverInput(MessagesState):
    """Class for specifying what the retriever input should be; can be used as a State class with LangGraph"""

    # it is not necessary for this to inherit from MessagesState if we're not using LangGraph
    # however, it allows the option in future and there is a neatness about it
    # RetrieverInput inherits a `messages` property from MessagesState
    limit: int
    filter_condition: str


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class CustomRetriever(BaseRetriever):
    """Custom retriever class because I encountered a bug when converting a LanceDB
    vector store into a retriever in the usual way"""  # noqa

    # async def _aget_relevant_documents(self, query: str, limit: int = 3, **kwargs) -> List[LangchainDocument]:
    # may not be needed
    # there have been problems getting Lance DB to work with asynchronous requests
    #    pass

    def _get_relevant_documents(self, input: RetrieverInput, **kwargs) -> List[LangchainDocument]:
        """
        Retrieve chunks related to a search query using a hybrid search strategy

        CAUTION: kwargs are not passed on when the retriever is part of a rag_chain and the rag_chain is invoked
        workaround – use modified create_retrieval_chain above and pass kwarg-like arguments via an input dict (rather than str)

        """

        logger.info(f"Input to retriever: {input}")

        db = lancedb.connect(DB_PATH)

        chunks = CustomRetriever.retrieve_chunks(db, input, **kwargs)

        if PROJECT == "NestaBrain":
            # Quick hack to give access for RAG to author and title information (by Karlis)
            for chunk in chunks:
                chunk.text = (
                    chunk.text + "; title: " + str(chunk.source.title) + "; authors: " + str(chunk.source.authors)
                )
            # (hack ends)

        docs = CustomRetriever.chunks_to_docs(chunks, enumerate_=True)

        return docs

    @staticmethod
    def chunks_to_docs(
        chunks: List[Chunk], merge: bool = default_merge, enumerate_: bool = False
    ) -> List[LangchainDocument]:
        """Convert Chunk objects to LangchainDocument objects, with the option to merge"""
        if merge:
            docs = CustomRetriever.merge_chunks(chunks, enumerate_=enumerate_)
            if len(docs) < len(chunks):
                logger.info(f"{len(chunks)} retreived chunks were merged into {len(docs)} chunks")
            return docs
        else:
            return [
                chunk.to_LangchainDocument(enumeration_index=i + 1 if enumerate_ else None)
                for i, chunk in enumerate(chunks)
            ]

    @staticmethod
    def merge_chunks(chunks: List[Chunk], enumerate_: bool = False) -> List[LangchainDocument]:
        """
        Identify which source document each chunk in a list of chunks is from.
        Then concatenate the texts of each chunk belonging to each individual document.
        Then create a new Langchain Document containing this concatenated text
        """  # noqa

        chunks_grouped_by_source = OrderedDict({})
        no_source_chunks = []  # chunks from a schema which does not have a source property

        for chunk in chunks:
            if hasattr(chunk, "source"):
                if chunk.source not in chunks_grouped_by_source:
                    chunks_grouped_by_source[chunk.source] = []
                chunks_grouped_by_source[chunk.source].append(chunk)
            else:
                no_source_chunks.append(chunk)

        docs = []
        for source, chunks in chunks_grouped_by_source.items():
            chunks = sorted(
                chunks, key=lambda chunk: chunk.order_index or 0
            )  # put the chunks in the order in which they appeared in the original document;
            # some chunks which were ingested initially will have order_index = None;
            # no chunk should lack an order_index if other chunks from the same document have one
            text = "\n\n".join([chunk.text for chunk in chunks])
            metadata = source.as_metadata()
            metadata.update({"schema": Chunk.__name__})
            doc = Chunk.to_LangchainDocument_(
                text, metadata=metadata, enumeration_index=len(docs) + 1 if enumerate_ else None
            )
            docs.append(doc)

        for chunk in no_source_chunks:
            metadata = chunk.metadata
            metadata.update({"schema": chunk.__class__.__name__})
            doc = chunk.to_LangchainDocument_(
                text, metadata=metadata, enumeration_index=len(docs) + 1 if enumerate_ else None
            )
            docs.append(doc)

        return docs

    @staticmethod
    def retrieve_chunks(
        db: LanceDBConnection,
        input: RetrieverInput,
        include_projects: bool = True,
        quantile_limit: float = 0.333,
        **kwargs,
    ) -> List[Chunk]:
        """Retrieve chunks synchrously"""

        chunk_table = db.open_table(chunk_table_name)
        if include_projects:
            project_table = db.open_table("mission_project")
            tables = [chunk_table, project_table]
        else:
            tables = [chunk_table]

        query = input["messages"][-1].content
        limit = input["limit"]
        filter_condition = input.get("filter_condition") or None  # if '' then want None
        logger.info("Vectorizing query ...")
        vector_ = CustomRetriever.vector(query)

        logger.info("Retrieving most relevant chunks ...")
        chunks = CustomRetriever.search_loop(
            tables, query, vector_, limit, filter_condition=filter_condition, **kwargs
        )
        ranked_chunks = sorted(chunks, key=lambda chunk: chunk.relevance_score, reverse=True)
        quantile = np.quantile([chunk.relevance_score for chunk in chunks], quantile_limit)
        top_chunks = [chunk for chunk in ranked_chunks if chunk.relevance_score >= quantile]
        chunks = top_chunks[0:limit]  # ranked_chunks[0:limit]
        logger.info(f"Retrieved {len(chunks)} chunks")

        return chunks

    @staticmethod
    def search_loop(
        table_or_tables: Union[LanceTable, List[LanceTable]],
        query: str,
        vector_: List[float],
        limit: int,
        filter_condition: Optional[str] = None,
    ) -> List[Chunk]:
        """Search LanceDB table, omit duplicate chunks, repeat the action until there are limit unique chunks (synchronous)"""

        if type(table_or_tables) is list:
            tables = table_or_tables
        else:
            tables = [table_or_tables]

        query = re.sub(r"\W+", " ", query)

        all_chunks = []
        for table in tables:

            chunk_class = table_name_to_schema_class_map[table.name]
            is_main_chunk_class = chunk_class is Chunk

            iteration_required = True
            while iteration_required:  # iteration only necessary if there are duplicates, for example,
                # some 'boilerplate' text from reports may be duplicated
                chunks = (
                    table.search(query_type="hybrid")
                    .vector(vector_)
                    .text(query)
                    .where(
                        filter_condition if is_main_chunk_class else None,
                        prefilter=True,
                    )
                    .limit(limit)
                )

                relevance_scores = chunks.to_arrow()["_relevance_score"]

                chunks = chunks.to_pydantic(chunk_class)
                for i, chunk in enumerate(chunks):
                    chunk.relevance_score = relevance_scores[
                        i
                    ].as_py()  # as_py converts a pyarrow.lib.FloatScalar to a float
                    chunk.use_as_context = is_main_chunk_class

                found_limit_chunks = len(chunks) == limit
                unique_chunks = unique(
                    chunks
                )  # there shouldn't be many duplicate chunks in the DB, but this removes the possibility of returning them
                chunks_arent_unique = len(unique_chunks) < len(chunks)
                iteration_required = found_limit_chunks and chunks_arent_unique
                if iteration_required:
                    limit = limit * 2  # may need to increase the limit if chunks weren't unique and try again

            all_chunks += unique_chunks

        return all_chunks

    @staticmethod
    def vector(string: str) -> List[float]:
        """Calculate the embedding vector of string"""
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        vector = client.embeddings.create(model=DEFAULT_EMBEDDINGS_MODEL, input=string).data[0].embedding
        return vector


if __name__ == "__main__":

    load_dotenv()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    db = lancedb.connect(DB_PATH)
    doc_table = db.open_table("document")
    chunk_table = db.open_table(chunk_table_name)
    project_table = db.open_table("mission_project")

    # code below is just for testing and experimenting

    if False:
        # experimenting with combining results and reranking

        import numpy as np
        import pyarrow as pa

        from lancedb.rerankers.rrf import RRFReranker

        query = "How do you design a Collective Intelligence Project?"
        vector_ = CustomRetriever.vector(query)
        reranker = RRFReranker()

        chunks = chunk_table.search(query_type="hybrid").vector(vector_).text(query).limit(10)

        projects = project_table.search(query_type="hybrid").vector(vector_).text(query).limit(10)

        all_relevance_scores = []
        results = []
        for i, query_result in enumerate([chunks, projects]):
            arrow_result = query_result.to_arrow()
            relevance_scores = [score.as_py() for score in arrow_result["_relevance_score"]]
            all_relevance_scores += relevance_scores
            pydantic_class = Chunk if i == 0 else MissionProject
            pydantic_result = query_result.to_pydantic(pydantic_class)
            for j, element in enumerate(pydantic_result):
                element.relevance_score = relevance_scores[j]
            results += pydantic_result

        # print(all_relevance_scores)

        if False:
            t1 = pa.table([chunks["text"], chunks["vector"]], names=["text", "vector"])
            t2 = pa.table([projects["text"], projects["vector"]], names=["text", "vector"])
            res = pa.concat_tables([t1, t2])
            v = pa.table([pa.array(["1", "2", "3", "4"]), res["vector"]], names=["_rowid", "vector"])
            t = pa.table([pa.array(["1", "2", "3", "4"]), res["text"]], names=["_rowid", "text"])
            res = reranker.rerank_multivector([v], query=query)
        #   print(res)

    if False:
        # testing search_loop with a different table
        query = "What work has Nesta done on educational technology"
        input = {"messages": [HumanMessage(content=query)], "limit": 1}
        res = CustomRetriever()._get_relevant_documents(input, append_projects=True)
    #    print(res)

    if False:
        # experimenting with search filter conditions
        query = "What work has Nesta done on educational technology"
        # query = 'Who has experience working in government'
        filter_condition = "source.date_pub >= to_timestamp('2020-01-01')"  # filter by date
        # filter_condition = "array_contains(source.projects,'Digital Arts and Culture Accelerator')" #filter by project.
        # Remember the 'projects' field is a list of strings (there can be more than one project)
        # filter_condition = "source.contentType = 'person page'"  #filter by content type
        # filter_condition = "source.rank <= 100" #filter by page popularity
        #  filter_condition = None  #also works with no filter condition
        filter_condition = "source.date_pub >= to_timestamp('2020-01-01') and source.contentType = 'person page'"
        chunks = CustomRetriever().invoke(query, filter_condition=filter_condition)
        logger.info(len(chunks))
        for chunk in chunks:
            logger.info("\n\n", chunk)

    if False:
        # experimenting with asynchronous search
        # see https://lancedb.github.io/lancedb/hybrid_search/hybrid_search/#hybrid-search-in-lancedb
        # Doesn't work
        async def foo():  # noqa
            query = "What work has Nesta done on educational technology"
            v = CustomRetriever.vector(query)
            async_db = await lancedb.connect_async(DB_PATH)
            async_tbl = await async_db.open_table("chunk")
            return await async_tbl.query().nearest_to(v).nearest_to_text(query).limit(1).to_pandas()

        res = asyncio.run(foo())
    #   print(res)

    if False:
        # experimenting with queries
        # lancedb's neater syntax for handling embeddings doesn't work because of the way the schema has been specified

        query = "HACID"
        #  vector_ = client.embeddings.create(model=DEFAULT_EMBEDDINGS_MODEL, input=query).data[0].embedding
        # chunk_table.create_fts_index("text")
        # chunk_results = chunk_table.search()
        #                   .where('source.location = "https://www.nesta.org.uk/project/centre-collective-intelligence-design/"')
        #                   .to_list()

        chunk_results = (
            chunk_table.search(query)
            .where("source.date_pub >= to_timestamp('2019-01-01')")
            .limit(3)
            .to_pydantic(Chunk)
        )

        for result in chunk_results:
            logger.info("\n\n", result, "\n\n")

    if False:
        # experimenting with Langchain

        vector_store = LanceDB(
            uri=DB_PATH,
            embedding=OpenAIEmbeddings(),
            table_name=chunk_table_name,
        )

        retriever = vector_store.as_retriever()
        docs = retriever.invoke("climate change projects")  # bug in lancedb prevents this from working
        logger.info(docs)
