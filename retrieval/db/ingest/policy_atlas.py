import argparse
import logging

import pandas as pd
import retrieval.db.ingest.const as const
import retrieval.db.ingest.ingest as ing

from dsp_nesta_brain import PROJECT_DIR
from dsp_nesta_brain import logger
from langchain.docstore.document import Document as LangchainDocument
from retrieval.db.schema.policy_atlas import Activity as Chunk


DATA_PATH = PROJECT_DIR / "data/policy_atlas/fcdo_iati_data_2025_01_17.csv"


def chunk_already_in_db(chunk: LangchainDocument, **kwargs) -> bool:
    """Determine whether identical chunks have already been added to the database.
    Chunking strategy should have been the same.
    """  # noqa

    where_condition = f'iati_identifier == "{chunk.metadata["iati_identifier"]}"'
    results = ing.chunk_already_in_db(chunk, where_condition=where_condition, **kwargs)
    return bool(results)


async def chunk_to_Chunk(chunk: LangchainDocument, ingestion: bool = True) -> Chunk:
    """
    Convert a Langchain document into an object
    of the Chunk class which can be ingested into the DB
    (including deriving an embedding for the Chunk)
    """  # noqa

    try:
        return await ing.chunk_to_Chunk(chunk, ingestion=ingestion, **chunk.metadata)
    except Exception as e:
        return e


# !!!!!!!!!!!!!!!!!!!!!!!!!!
# documents_to_Chunks
# AND
# ingest
# methods were at one point deleted from this file  (when compared with handover-tidying-0)
# Assume they are not needed anymore?
# CHECK THIS MAKES SENSE
# !!!!!!!!!!!!!!!!!!!!!!!!!!!

if __name__ == "__main__":

    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_index", type=int, default=0
    )  # the row of the policy data CSV to start ingesting; everything prior to this will be ignored
    parser.add_argument(
        "--batch_size", type=int, default=10
    )  # the number of CSV rows to ingest at a time. Batch sizes of 230+ seem to get errors back from OpenAI.

    # settings constants which may be needed in other files
    const.Chunk = Chunk
    const.CHUNK_TABLE_NAME = "activity"

    args = parser.parse_args()

    data = pd.read_csv(DATA_PATH)
    N_rows = data.shape[0]

    for start_index_ in range(args.start_index, N_rows, args.batch_size):

        logger.info(f"Ingesting records {start_index_} to {start_index_ + args.batch_size - 1} ...")
        ing.csv_rows_to_ingested_data(
            DATA_PATH,
            start_index_,
            args.batch_size,
            #  identifier="iati_identifier",
            Chunk_func=chunk_to_Chunk,
            chunk_presence_test=chunk_already_in_db,
        )
