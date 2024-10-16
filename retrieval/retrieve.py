import os

from collections import OrderedDict
from typing import List
from typing import Union

import lancedb

from dotenv import load_dotenv
from dsp_nesta_brain import logger
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from retrieval.db.schema import Chunk


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


def vector(string: str) -> List[float]:
    """Calculate the embedding vector of string"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    vector = client.embeddings.create(model="text-embedding-3-small", input=string).data[0].embedding
    return vector


def chunks_to_langchain_documents(chunks: List[Chunk]) -> List[LangchainDocument]:
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

    langchain_documents = []
    for source, chunks in chunks_grouped_by_source.items():
        text = "\n\n".join(
            [chunk.text for chunk in chunks]
        )  # chunks aren't necessarily in the right order – can add an order number at the time of ingestion to fix this
        langchain_document = LangchainDocument(page_content=text, metadata=source.as_metadata())
        langchain_documents.append(langchain_document)

    return langchain_documents


def retrieve(
    query: str, limit: int = 3, as_langchain_document: bool = True, merge: bool = False
) -> Union[List[Chunk], List[LangchainDocument]]:
    """
    Retrieve chunks related to a search query using a hybrid search strategy

    as_langchain_document: if True then retrieval results are Langchain Documents rather than Chunks
    merge: if True then where chunks are from the same document they will be merged into a single retrieval result
    """

    if merge and not as_langchain_document:
        logger.warning("Cannot merge retrieved results if as_langchain_document is not True")
        merge = False

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    db = lancedb.connect("retrieval/db/ccid_demo_db")
    chunk_table = db.open_table("chunk")

    logger.info("Vectorizing query ...")
    vector_ = vector(query)

    logger.info("Retrieving most relevant chunks ...")
    results = []
    orig_limit = limit
    while len(results) < orig_limit:  # only necessary if there are duplicates (which there shouldn't be)
        results += chunk_table.search(query_type="hybrid").vector(vector_).text(query).limit(limit).to_pydantic(Chunk)
        results = unique(
            results
        )  # there shouldn't be duplicate chunks in the DB, but this removes the possibility of returning them
        limit = limit * 2  # increase the limit if results weren't unique and try again

    results = results[0:orig_limit]
    N_chunks = len(results)
    logger.info(f"Retreived {N_chunks} chunks")

    if as_langchain_document:
        if merge:
            results = chunks_to_langchain_documents(results)
            logger.info(f"{N_chunks} retreived chunks were merged into {len(results)} chunks")
        else:
            results = [chunk.to_LangchainDocument() for chunk in results]

    return results


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
