import logging
import sys

from typing import List

import lancedb
import pandas as pd
import retrieval.db.ingest.const as const
import retrieval.db.ingest.ingest as ing

from dsp_nesta_brain import PROJECT_DIR
from dsp_nesta_brain import logger
from langchain.docstore.document import Document as LangchainDocument
from retrieval.db.schema.policy_atlas import Activity


DATA_PATH = PROJECT_DIR / "data/policy_atlas/fcdo_iati_data_2025_01_17.csv"


def chunk_already_in_db(chunk: LangchainDocument, **kwargs) -> bool:
    """Determine whether identical chunks have already been added to the database.

    Chunking strategy should have been the same.
    """

    where_condition = f'iati_identifier == "{chunk.metadata["iati_identifier"]}"'
    results = ing.chunk_already_in_db(chunk, where_condition=where_condition, **kwargs)
    return bool(results)


async def chunk_to_Chunk(chunk: LangchainDocument, ingestion: bool = True) -> Activity:
    """
    Convert a Langchain document into an object
    of the Chunk class which can be ingested into the DB
    (including deriving an embedding for the Chunk)
    """  # noqa

    try:
        return await ing.chunk_to_Chunk(chunk, ingestion=ingestion, **chunk.metadata)
    except Exception as e:
        return e


if __name__ == "__main__":

    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    # SETTINGS
    start_index = (
        int(sys.argv[1]) if len(sys.argv) > 1 else 0
    )  # the row of the policy data CSV to start ingesting; everything prior to this will be ignored
    batch_size = (
        225  # the number of CSV rows to ingest at a time. Batch sizes of 230+ seem to get errors back from OpenAI.
    )

    # settings constants which may be needed in other files
    const.Chunk = Activity
    const.CHUNK_TABLE_NAME = "activity"

    # global variable
    request_counter = ing.RequestCounter()

    data = pd.read_csv(DATA_PATH)
    N_rows = data.shape[0]

    for start_index_ in range(start_index, N_rows, batch_size):

        logger.info(f"Ingesting records {start_index_} to {start_index_ + batch_size - 1} ...")
        ing.csv_rows_to_ingested_data(
            DATA_PATH,
            start_index_,
            batch_size,
            identifier="iati_identifier",
            Chunk_func=chunk_to_Chunk,
            chunk_presence_test=chunk_already_in_db,
        )
