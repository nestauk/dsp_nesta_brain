from typing import Literal


PROJECT: Literal["NESTA_BRAIN", "POLICY_ATLAS"] = "POLICY_ATLAS"

if PROJECT == "NESTA_BRAIN":
    DB_PATH = "retrieval/db/full_site_demo_db_with_pdfs"
    DEFAULT_MODEL = "gpt-4o-mini"
    EARLIEST_YEAR = 2003  # 2003 is the earliest publication date in the DB
    DEFAULT_START_YEAR = 2019
