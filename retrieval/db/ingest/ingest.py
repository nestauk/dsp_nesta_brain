from __future__ import annotations

import asyncio
import logging
import math
import os

from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import lancedb
import pandas as pd
import retrieval.db.ingest.const as const

from config import DB_PATH
from dotenv import load_dotenv
from dsp_nesta_brain import logger
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import CharacterTextSplitter
from retrieval.embeddings import vector


DB = lancedb.connect(DB_PATH)
CHUNK_TABLE = DB.open_table(
    const.CHUNK_TABLE_NAME
)  # this should be set in retrieval/db/ingest/nesta_brain.py or retrieval/db/ingest/policy_atlas.py

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def standardise_text(text: str) -> str:
    """Standardise text for embedding purposes."""
    return text.replace('"', "”")  # avoid Unterminated string literal errors


def chunk_already_in_db(chunk: LangchainDocument, where_condition: Optional[str] = None) -> bool:
    """Determine whether identical chunks have already been added to the database, because PDFs may be duplicated across the site.
    The chunking strategy needs to have been the same for this to work.
    """  # noqa

    where_condition = where_condition or f'text == """{standardise_text(chunk.page_content)}"""'
    try:
        results = CHUNK_TABLE.search().where(where_condition).limit(1).to_pydantic(const.Chunk)
    except Exception as e:
        error_message = "Error while trying to check whether a chunk exists in the database"
        logger.error(error_message)
        raise Exception(e)
    return results


async def chunk_to_Chunk(chunk: LangchainDocument, **kwargs) -> const.Chunk:
    """
    Convert a Langchain chunk (a section of text from a document, possibly returned from a text splitter) into an object
    of the Chunk class as defined by the DB schema which can be ingested into the DB
    (including deriving an embedding for the Chunk)
    """  # noqa
    vector_ = await vector(standardise_text(chunk.page_content), async_=True)
    return const.Chunk(text=chunk.page_content, vector=vector_, **kwargs)


def split_documents(documents: List[LangchainDocument]) -> List[LangchainDocument]:
    """Split documents into chunks"""

    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs_split = text_splitter.split_documents(documents)

    if documents and not docs_split:
        raise Exception(f"Investigate why you have zero chunks for {len(documents)} documents")

    return docs_split


async def documents_to_Chunks_no_split(
    documents: List[LangchainDocument],
    skip_message_format: Optional[str] = None,
    Chunk_func: Callable = chunk_to_Chunk,
) -> List[const.Chunk]:
    """Convert LangchainDocuments into objects of the Chunk class (without splitting them) which can be ingested into the DB"""

    def log_exceptions(task_results: List[Union[const.Chunk, Exception]]) -> None:

        message_format = 'Task {index} raised an exception "{exception}" within asyncio.gather'
        exceptions = [(i, ele) for i, ele in enumerate(task_results) if isinstance(ele, Exception)]

        for index, exception in exceptions:
            message = message_format.format(index=index, exception=str(exception))
            logger.error(message)

        if exceptions:
            raise Exception("Exceptions in documents_to_Chunks_no_split")

    chunks = []
    for doc in documents:

        if chunk_already_in_db(doc):

            if skip_message_format:
                logger.info(skip_message_format.format(**doc.metadata))

        else:
            chunks.append(doc)

    if chunks:
        tasks = [asyncio.create_task(Chunk_func(chunk)) for chunk in chunks]
        logger.info(f"Fetching embeddings for {len(chunks)} chunks ...")
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)
        log_exceptions(gather_results)
        return gather_results

    return []


def ingest(documents: List[LangchainDocument], **kwargs) -> None:
    """
    Find out which documents are not already in the database, convert them into
    Chunk data in accordance with the db schema
    and insert this into the database
    """  # noqa

    logging.getLogger("httpx").setLevel(logging.WARNING)

    chunks = asyncio.run(documents_to_Chunks_no_split(documents, **kwargs))

    if chunks:
        logger.info(f"Ingested {len(chunks)} Chunks into the database")
        CHUNK_TABLE.add(chunks)
    else:
        logger.info("No chunks to ingest into the database")

    logging.getLogger("httpx").setLevel(logging.INFO)


def csv_rows_to_ingested_data(
    path: str, start_index: int, batch_size: Union[int, None], text_col: Union[str, List[str]] = "text", **kwargs
) -> None:
    """Ingest data from the rows of a CSV file"""

    text_col = text_col if type(text_col) is list else [text_col]

    data = pd.read_csv(path)
    rows = data[start_index : (start_index + batch_size) if batch_size else None].to_dict(orient="records")

    docs = []
    for row in rows:

        text = ""
        for text_col_name in text_col:
            text_component = row.get(text_col_name)
            if (type(text_component) is float and math.isnan(text_component)) or str(text_component).lower() == "nan":
                text_component = ""
            if text_component:
                text += " " + text_component

        doc = LangchainDocument(page_content=text, metadata=row)
        docs.append(doc)

    ingest(docs, **kwargs)
