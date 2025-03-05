from typing import Literal


PROJECT: Literal["NESTA_BRAIN", "POLICY_ATLAS"] = "NESTA_BRAIN"

DEFAULT_MODEL: str = "gpt-4o-mini"
DEFAULT_EMBEDDINGS_MODEL: str = "text-embedding-3-small"

DEBUG_MODE = False

USE_AZURE: bool = True
USE_AZURE_LLM: bool = USE_AZURE
AZURE_MODEL: str = DEFAULT_MODEL
AZURE_API_VERSION: str = "2024-10-21"
USE_AZURE_EMBEDDINGS: bool = USE_AZURE
AZURE_EMBEDDINGS_MODEL: str = DEFAULT_EMBEDDINGS_MODEL

if PROJECT == "NESTA_BRAIN":
    DB_PATH: str = "retrieval/db/nesta_brain"
    EARLIEST_YEAR: int = 2003  # 2003 is the earliest publication date in the DB
    DEFAULT_START_YEAR: int = 2019
    USE_LANGFUSE: bool = True  # set as wanted

if PROJECT == "POLICY_ATLAS":
    DB_PATH: str = "retrieval/db/policy_atlas"
    EARLIEST_YEAR: int = 1900
    DEFAULT_START_YEAR: int = 2019
    USE_LANGFUSE: bool = False  # Langfuse is not currently set up for other projects –
    # don't want NestaBrain's Langfuse to store traces from other projects
