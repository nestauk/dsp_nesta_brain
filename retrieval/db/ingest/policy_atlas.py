import asyncio
import logging
import sys

from typing import List

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

    return await ing.chunk_to_Chunk(chunk, ingestion=ingestion, **chunk.metadata)


async def documents_to_Chunks(documents: List[LangchainDocument]) -> List[Chunk]:
    """
    Split Langchain documents into chunks and convert these into objects
    of the Chunk class which can be ingested into the DB
    """  # noqa

    logger.info(f"Fetching embeddings for {len(documents)} chunks ...")
    tasks = []
    for chunk in documents:  # the variable name 'chunk' is possibly a bit misleading here.
        # There should be no need to split documents into chunks as activity texts aren't long enough

        if chunk_already_in_db(chunk):

            logger.info(f"Skipping chunk {repr(chunk)} as it already seems to be in the DB")

        else:
            task = asyncio.create_task(chunk_to_Chunk(chunk))
            tasks.append(task)

    await ing.throttle([chunk.page_content for chunk in documents])
    return await asyncio.gather(*tasks)


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

    # SETTINGS
    start_index = (
        int(sys.argv[1]) if len(sys.argv) > 1 else 0
    )  # the row of the policy data CSV to start ingesting; everything prior to this will be ignored
    batch_size = 100  # the number of CSV rows to ingest at a time

    csv_rows_to_ingested_data(start_index, batch_size)
