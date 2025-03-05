from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from typing import List

import lancedb
import pandas as pd

from config import DB_PATH
from google_drive.drive_doc import BaseDriveDoc
from pydantic import Field
from retrieval.db.schema.nesta_brain import Chunk
from retrieval.db.schema.nesta_brain import Document as LanceDocument


class Policy(BaseDriveDoc):
    """A class for describing Nesta policy documents."""

    file_id: str = (
        Field(
            ...,
            description="A unique identifier",
        ),
    )
    title: str = (
        Field(
            ...,
            description="The title of the document and the policy it covers",
        ),
    )
    date_pub: str = (
        Field(
            ...,
            description="The date the policy was published",
        ),
    )

    def __init__(self, document: LanceDocument) -> None:
        """Initialize the class with the document object"""
        date_pub = datetime.strftime(document.date_pub, "%B %Y")
        super().__init__(file_id=document.file_id, title=document.title, date_pub=date_pub)

    @staticmethod
    def list(as_string: bool = False, to_csv: bool = False) -> List[Policy]:
        """Get all the policies in the vector DB and convert to the Policy class"""

        db = lancedb.connect(DB_PATH)

        chunk_table = db.open_table(
            "chunk"
        )  # using the chunk table instead of the document table in case the latter is deleted

        chunks = chunk_table.search().where('source.drive_type = "policy"').limit(-1).to_pydantic(Chunk)

        policies = [Policy(chunk.source) for chunk in chunks]

        if to_csv:
            df = pd.DataFrame.from_records([policy.to_dict() for policy in policies])
            df.to_csv("retrieval/db/ingest/policies.csv", index=False)

        if as_string:
            return "\n".join([repr(policy) for policy in policies])

        else:
            return policies

    def to_dict(self) -> OrderedDict:
        """Return fields as an ordered dictionary, for example, for conversion into a row in a dataframe"""

        dict_ = self.__dict__
        dict_["link"] = f"https://drive.google.com/file/d/{dict_['file_id']}"
        return OrderedDict({k: dict_[k] for k in ["title", "date_pub", "link"]})
