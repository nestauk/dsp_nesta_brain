from __future__ import annotations

import asyncio
import os

from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import lancedb

from config import DB_PATH
from config import PROJECT
from dotenv import load_dotenv
from dsp_nesta_brain import logger
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import CharacterTextSplitter
from retrieval.db.schema.nesta_brain import Chunk as NestaBrainChunk
from retrieval.db.schema.policy_atlas import Activity
from retrieval.embeddings import vector


# the definition of Chunk and chunk_table_name may depend on settings in other files
if PROJECT == "NESTA_BRAIN":
    Chunk = NestaBrainChunk
    CHUNK_TABLE_NAME = "chunk"
elif PROJECT == "POLICY_ATLAS":
    Chunk = Activity
    CHUNK_TABLE_NAME = "activity"

DB = lancedb.connect(DB_PATH)
CHUNK_TABLE = DB.open_table(CHUNK_TABLE_NAME)

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
        results = CHUNK_TABLE.search().where(where_condition).limit(1).to_pydantic(Chunk)
    except Exception as e:
        error_message = "Error while trying to check whether a chunk exists in the database"
        logger.error(error_message)
        raise Exception(e)
    return results


async def chunk_to_Chunk(chunk: LangchainDocument, **kwargs) -> Chunk:
    """
    Convert a Langchain chunk (a section of text from a document, possibly returned from a text splitter) into an object
    of the Chunk class as defined by the DB schema which can be ingested into the DB
    (including deriving an embedding for the Chunk)
    """  # noqa
    vector_ = await vector(standardise_text(chunk.page_content), async_=True)
    return Chunk(text=chunk.page_content, vector=vector_, **kwargs)


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
    chunk_to_Chunk: Callable = chunk_to_Chunk,
) -> List[Chunk]:
    """Convert LangchainDocuments into objects of the Chunk class (without splitting them) which can be ingested into the DB"""

    def log_exceptions(task_results: List[Union[Chunk, Exception]]) -> None:

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
        tasks = [asyncio.create_task(chunk_to_Chunk(chunk)) for chunk in chunks]
        logger.info(f"Fetching embeddings for {len(chunks)} chunks ...")
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)
        log_exceptions(gather_results)
        return gather_results

    return []
