from typing import Literal


PROJECT: Literal["NESTA_BRAIN", "POLICY_ATLAS"] = "POLICY_ATLAS"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDINGS_MODEL = "text-embedding-3-small"

if PROJECT == "NESTA_BRAIN":
    DB_PATH = "retrieval/db/full_site_demo_db_with_pdfs"
    EARLIEST_YEAR = 2003  # 2003 is the earliest publication date in the DB
    DEFAULT_START_YEAR = 2019

if PROJECT == "POLICY_ATLAS":
    DB_PATH = "retrieval/db/policy_atlas"
