import asyncio
import logging
import sys
import traceback

from typing import List
from typing import Union

import lancedb
import pandas as pd
import retrieval.db.ingest.ingest as ing

from config import DB_PATH
from dsp_nesta_brain import PROJECT_DIR
from dsp_nesta_brain import logger
from langchain.docstore.document import Document as LangchainDocument
from retrieval.db.schema.policy_atlas import Activity as Chunk


DATA_PATH = PROJECT_DIR / "data/policy_atlas/fcdo_iati_data_2025_01_17.csv"

db = lancedb.connect(DB_PATH)
chunk_table = db.open_table("activity")


def chunk_already_in_db(chunk: LangchainDocument) -> bool:
    """Determine whether identical chunks have already been added to the database.
    Chunking strategy should have been the same.
    """  # noqa

    where_condition = f'iati_identifier == "{chunk.metadata["iati_identifier"]}"'
    results = ing.chunk_already_in_db(chunk, where_condition=where_condition)
    return bool(results)


async def chunk_to_Chunk(chunk: LangchainDocument, ingestion: bool = True) -> Chunk:
    """
    Convert a Langchain chunk (as returned from a text splitter) into an object
    of the Chunk class which can be ingested into the DB
    (including deriving an embedding for the Chunk)
    """  # noqa

    try:
        return await ing.chunk_to_Chunk(chunk, ingestion=ingestion, **chunk.metadata)
    except Exception as e:
        return e


async def documents_to_Chunks(documents: List[LangchainDocument]) -> List[Chunk]:
    """
    Split Langchain documents into chunks and convert these into objects
    of the Chunk class which can be ingested into the DB
    """  # noqa

    def log_exceptions(task_results: List[Union[Chunk, Exception]]) -> None:

        message_format = (
            'Task {index} raised an exception "{exception}" within asyncio.gather. \tTraceback:\n\t{traceback}'
        )
        exceptions = [(i, ele) for i, ele in enumerate(task_results) if isinstance(ele, Exception)]

        for i, exception in enumerate(exceptions):
            message = message_format.format(
                index=i, exception=str(exception), traceback=traceback.format_tb(exception.__traceback__)
            )
            logging.error(message)

        if exceptions:
            raise Exception("Exceptions in documents_to_Chunks")

    logger.info(f"Fetching embeddings for {len(documents)} chunks ...")
    tasks = []
    for chunk in documents:  # the variable name 'chunk' is possibly a bit misleading here.
        # There should be no need to split documents into chunks as activity texts aren't long enough

        if chunk_already_in_db(chunk):

            logger.info(f"Skipping chunk {chunk.metadata.get('iati_identifier')} as it already seems to be in the DB")

        else:
            task = asyncio.create_task(chunk_to_Chunk(chunk))
            tasks.append(task)

    if tasks:
        await ing.throttle([chunk.page_content for chunk in documents])
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)
        log_exceptions(gather_results)
        return gather_results

    return []


def ingest(documents: List[LangchainDocument], replace: bool = False) -> None:
    """
    Find out which documents are not already in the database, convert them into
    Document and Chunk data in accordance with the db schema
    and insert this into the database
    """  # noqa

    logging.getLogger("httpx").setLevel(logging.WARNING)

    chunks = asyncio.run(documents_to_Chunks(documents))

    if chunks:
        logger.info(f"Ingested {len(chunks)} Chunks into the database")
        chunk_table.add(chunks)
    else:
        logger.info("No chunks to ingest into the database")

    logging.getLogger("httpx").setLevel(logging.INFO)


def csv_rows_to_ingested_data(start_index: int, batch_size: int) -> None:
    """Perform a search, scrape the webpages from the search results, and ingest the data"""

    data = pd.read_csv(DATA_PATH)
    rows = data[start_index : (start_index + batch_size)].to_dict(orient="records")

    docs = [LangchainDocument(page_content=row.pop("text"), metadata=row) for row in rows]

    ingest(docs)


if __name__ == "__main__":

    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    # SETTINGS
    start_index = (
        int(sys.argv[1]) if len(sys.argv) > 1 else 0
    )  # the row of the policy data CSV to start ingesting; everything prior to this will be ignored
    batch_size = (
        200  # the number of CSV rows to ingest at a time. Batch sizes of 250+ seem to get errors back from OpenAI.
    )

    data = pd.read_csv(DATA_PATH)
    N_rows = data.shape[0]

    for start_index_ in range(start_index, N_rows, batch_size):

        logger.info(f"Ingesting records {start_index_} to {start_index_ + batch_size - 1} ...")
        csv_rows_to_ingested_data(start_index_, batch_size)
