from typing import Type


# this file contains variables which are dependent on project settings but used in more than one module
# see #retrieval/db/ingest/nesta_brain.py and retrieval/db/ingest/policy_atlas.py for how they are set
# also used in retrieval/db/ingest/ingest.py
Chunk: Type = None  # the schema class to interpret as a "Chunk" in the database
CHUNK_TABLE_NAME: str = None  # the name of the Lance DB table containing the chunks
