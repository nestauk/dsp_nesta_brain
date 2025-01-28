from __future__ import annotations

import os
import re

from collections import OrderedDict
from typing import List
from typing import Optional

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
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import MessagesState
from openai import OpenAI
from retrieval.db.schema.nesta_brain import Chunk as NestaBrainChunk
from retrieval.db.schema.policy_atlas import Activity
from utils import unique


if PROJECT == "NESTA_BRAIN":
    Chunk = NestaBrainChunk
    chunk_table_name = "chunk"
    default_merge = True
elif PROJECT == "POLICY_ATLAS":
    Chunk = Activity
    chunk_table_name = "activity"
    default_merge = False  # activity records were not split into separate chunks,so no need to merge


class RetrieverInput(MessagesState):
    """Class for specifying what the retriever input should be; used as a State class with LangGraph"""

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
        for chunk in chunks:
            if chunk.source not in chunks_grouped_by_source:
                chunks_grouped_by_source[chunk.source] = []
            chunks_grouped_by_source[chunk.source].append(chunk)

        docs = []
        for source, chunks in chunks_grouped_by_source.items():
            chunks = sorted(
                chunks, key=lambda chunk: chunk.order_index or 0
            )  # put the chunks in the order in which they appeared in the original document;
            # some chunks which were ingested initially will have order_index = None;
            # no chunk should lack an order_index if other chunks from the same document have one
            text = "\n\n".join([chunk.text for chunk in chunks])
            doc = Chunk.to_LangchainDocument_(
                text, metadata=source.as_metadata(), enumeration_index=len(docs) + 1 if enumerate_ else None
            )
            docs.append(doc)

        return docs

    @staticmethod
    def retrieve_chunks(db: LanceDBConnection, input: RetrieverInput, **kwargs) -> List[Chunk]:
        """Retrieve chunks synchrously"""

        chunk_table = db.open_table(chunk_table_name)

        query = input["messages"][-1].content
        limit = input["limit"]
        filter_condition = input.get("filter_condition") or None  # if '' then want None
        logger.info("Vectorizing query ...")
        vector_ = CustomRetriever.vector(query)

        logger.info("Retrieving most relevant chunks ...")
        chunks = CustomRetriever.search_loop(
            chunk_table, query, vector_, limit, filter_condition=filter_condition, **kwargs
        )
        chunks = chunks[0:limit]
        logger.info(f"Retreived {len(chunks)} chunks")
        return chunks

    @staticmethod
    def search_loop(
        table: LanceTable, query: str, vector_: List[float], limit: int, filter_condition: Optional[str] = None
    ) -> List[Chunk]:
        """Search LanceDB table, omit duplicate chunks, repeat the action until there are limit unique chunks (synchronous)"""

        query = re.sub(r"\W+", " ", query)
        iteration_required = True
        while iteration_required:  # iteration only necessary if there are duplicates, for example,
            # some 'boilerplate' text from reports may be duplicated
            chunks = (
                table.search(query_type="hybrid")
                .vector(vector_)
                .text(query)
                .where(
                    filter_condition,
                    prefilter=True,
                )
                .limit(limit)
                .to_pydantic(Chunk)
            )
            found_limit_chunks = len(chunks) == limit
            unique_chunks = unique(
                chunks
            )  # there shouldn't be many duplicate chunks in the DB, but this removes the possibility of returning them
            chunks_arent_unique = len(unique_chunks) < len(chunks)
            iteration_required = found_limit_chunks and chunks_arent_unique
            if iteration_required:
                limit = limit * 2  # may need to increase the limit if chunks weren't unique and try again
            else:
                chunks = unique_chunks
        return chunks

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

    # code below is just for testing and experimenting

    if True:
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
        # experimenting with queries
        # lancedb's neater syntax for handling embeddings doesn't work because of the way the schema has been specified

        query = "HACID project"
        vector_ = client.embeddings.create(model=DEFAULT_EMBEDDINGS_MODEL, input=query).data[0].embedding
        # chunk_table.create_fts_index("text")
        # chunk_results = chunk_table.search()
        #                   .where('source.location = "https://www.nesta.org.uk/project/centre-collective-intelligence-design/"')
        #                   .to_list()

        chunk_results = chunk_table.search(query_type="hybrid").vector(vector_).text(query).limit(3).to_pydantic(Chunk)

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
