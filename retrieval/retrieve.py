import os

from collections import OrderedDict
from typing import List

import lancedb

from dotenv import load_dotenv
from dsp_nesta_brain import logger
from lancedb.db import LanceDBConnection
from lancedb.table import LanceTable
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import LanceDB
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from openai import OpenAI
from retrieval.db.schema import Chunk


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def unique(chunks: List[Chunk]) -> List[Chunk]:
    """Find chunks with unique text and retain order"""
    seen = {}
    result = []
    for item in chunks:
        if item.text in seen:
            continue
        seen[item.text] = 1
        result.append(item)
    return result


class CustomRetriever(BaseRetriever):
    """Custom retriever class because I encountered a bug when converting a LanceDB
    vector store into a retriever in the usual way"""  # noqa

    # code previously used for retrieval has been reused to create a formal CustomRetriever class

    async def _aget_relevant_documents(
        self, query: str, limit: int = 3, merge: bool = False
    ) -> List[LangchainDocument]:
        """
        Retrieve chunks related to a search query using a hybrid search strategy

        merge: if True then where chunks are from the same document they will be merged into a single retrieval result
        """

        #  async_db = await lancedb.connect_async("retrieval/db/ccid_demo_db")
        db = lancedb.connect("retrieval/db/ccid_demo_db")

        logger.info("Vectorizing query ...")
        vector_ = await CustomRetriever.async_vector(query)
        #   chunks = await CustomRetriever.async_retrieve_chunks(db,query,vector_,limit)
        chunks = CustomRetriever.retrieve_chunks(db, query, vector_, limit)
        docs = CustomRetriever.chunks_to_docs(chunks, merge=merge)

        return docs

    def _get_relevant_documents(self, query: str, limit: int = 10, merge: bool = False) -> List[LangchainDocument]:
        """
        Retrieve chunks related to a search query using a hybrid search strategy

        merge: if True then where chunks are from the same document they will be merged into a single retrieval result
        """

        # the code has been chopped up into bits which can be reused easily in both synchronous and asynchronous versions

        db = lancedb.connect("retrieval/db/ccid_demo_db")

        logger.info("Vectorizing query ...")
        vector_ = CustomRetriever.vector(query)
        chunks = CustomRetriever.retrieve_chunks(db, query, vector_, limit)
        docs = CustomRetriever.chunks_to_docs(chunks, merge=merge)

        return docs

    @staticmethod
    def chunks_to_docs(chunks: List[Chunk], merge: bool = False) -> List[LangchainDocument]:
        """Convert Chunk objects to LangchainDocument objects, with the option to merge"""
        if merge:
            docs = CustomRetriever.merge_chunks(chunks)
            logger.info(f"{len(chunks)} retreived chunks were merged into {len(docs)} chunks")
            return docs
        else:
            return [chunk.to_LangchainDocument() for chunk in chunks]

    @staticmethod
    def merge_chunks(chunks: List[Chunk]) -> List[LangchainDocument]:
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
            text = "\n\n".join(
                [chunk.text for chunk in chunks]
            )  # chunks aren't necessarily in the right order – can add an order number at the time of ingestion to fix this
            doc = LangchainDocument(page_content=text, metadata=source.as_metadata())
            docs.append(doc)

        return docs

    @staticmethod
    async def async_retrieve_chunks(
        db: LanceDBConnection, query: str, vector_: List[float], limit: int
    ) -> List[Chunk]:
        """Retrieve chunks asynchroously"""
        chunk_table = await db.open_table("chunk")
        logger.info("Retrieving most relevant chunks ...")
        chunks = await CustomRetriever.async_search_loop(chunk_table, query, vector_, limit)
        chunks = chunks[0:limit]
        logger.info(f"Retreived {len(chunks)} chunks")
        return chunks

    @staticmethod
    def retrieve_chunks(db: LanceDBConnection, query: str, vector_: List[float], limit: int) -> List[Chunk]:
        """Retrieve chunks synchroously"""
        chunk_table = db.open_table("chunk")
        logger.info("Retrieving most relevant chunks ...")
        chunks = CustomRetriever.search_loop(chunk_table, query, vector_, limit)
        chunks = chunks[0:limit]
        logger.info(f"Retreived {len(chunks)} chunks")
        return chunks

    @staticmethod
    async def async_search_loop(table: LanceTable, query: str, vector_: List[float], limit: int) -> List[Chunk]:
        """Search LanceDB table, omit duplicate chunks, repeat the action until there are
        limit unique chunks (should be asynchronous – see comment below)"""  # noqa
        chunks = []
        orig_limit = limit
        while len(chunks) < orig_limit:  # only necessary if there are duplicates (which there shouldn't be)
            # THIS DOESN'T WORK: #I can't see a way of doing asynchronous hybrid search at the moment
            chunks += (
                await table.search(query_type="hybrid").vector(vector_).text(query).limit(limit).to_pydantic(Chunk)
            )
            chunks = unique(
                chunks
            )  # there shouldn't be duplicate chunks in the DB, but this removes the possibility of returning them
            limit = limit * 2  # increase the limit if chunks weren't unique and try again
        return chunks

    @staticmethod
    def search_loop(table: LanceTable, query: str, vector_: List[float], limit: int) -> List[Chunk]:
        """Search LanceDB table, omit duplicate chunks, repeat the action until there are limit unique chunks (synchronous)"""
        chunks = []
        orig_limit = limit
        while len(chunks) < orig_limit:  # only necessary if there are duplicates (which there shouldn't be)
            chunks += table.search(query_type="hybrid").vector(vector_).text(query).limit(limit).to_pydantic(Chunk)
            chunks = unique(
                chunks
            )  # there shouldn't be duplicate chunks in the DB, but this removes the possibility of returning them
            limit = limit * 2  # increase the limit if chunks weren't unique and try again
        return chunks

    @staticmethod
    async def async_vector(string: str) -> List[float]:
        """Calculate the embedding vector of string"""
        async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        result = await async_client.embeddings.create(model="text-embedding-3-small", input=string)
        vector = result.data[0].embedding
        return vector

    @staticmethod
    def vector(string: str) -> List[float]:
        """Calculate the embedding vector of string"""
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        vector = client.embeddings.create(model="text-embedding-3-small", input=string).data[0].embedding
        return vector


if __name__ == "main":

    load_dotenv()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    db = lancedb.connect("retrieval/db/ccid_demo_db")
    doc_table = db.open_table("document")
    chunk_table = db.open_table("chunk")

    # code below is just for testing and experimenting

    if False:
        # experimenting with queries
        # lancedb's neater syntax for handling embeddings doesn't work because of the way the schema has been specified

        query = "HACID project"
        vector_ = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
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
            uri="retrieval/db/ccid_demo_db",
            embedding=OpenAIEmbeddings(),
            table_name="chunk",
        )

        retriever = vector_store.as_retriever()
        docs = retriever.invoke("climate change projects")  # bug in lancedb prevents this from working
        logger.info(docs)
