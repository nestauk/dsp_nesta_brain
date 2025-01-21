from __future__ import annotations

from datetime import datetime
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

import lancedb

from config import DB_PATH
from retrieval.db.schema.base import BaseChunk


class Activity(BaseChunk):
    """Defines the fields which an Activity Record contains in the LanceDB database"""

    # selected IATI fields
    iati_identifier: str
    title_narrative: str
    reporting_org_ref: str
    reporting_org_narrative: str
    participating_org_ref: List[str]
    participating_org_narrative: List[str]
    recipient_country_code: str
    sector_narrative: str
    description_narrative: str
    activity_status_code: int
    activity_date_iso_date: List[datetime]
    activity_date_type: List[int]
    activity_date_narrative: Optional[str]
    document_link_url: str
    document_link_title_narrative: str
    document_link_description_narrative: str
    min_date: datetime
    max_date: datetime
    min_year: int
    max_year: int
    activity_status: Literal[
        "Pipeline/identification", "Implementation", "Finalisation", "Closed", "Cancelled", "Suspended"
    ]

    order_index: int  # not Optional for Activity
    time_added: datetime  # not Optional for Activity

    def __eq__(self, other: object) -> bool:
        """Self-explanatory"""
        if not isinstance(other, Activity):
            return False
        return self.iati_identifier == other.iati_identifier

    def __hash__(self) -> int:
        """Self-explanatory"""
        return hash(self.iati_identifier)

    @property
    def metadata(self) -> Dict:
        """Activity metadata"""
        metadata = self.__dict__
        #  print("Check metadata:", metadata)
        # input()
        return metadata

    @property
    def text(self) -> str:
        """Return text field dynamically rather than duplicating data"""
        return self.title + " " + self.description_narrative


if __name__ == "__main__":

    # creata a database
    db = lancedb.connect(DB_PATH)

    # creating tables
    if True:
        table = db.create_table("activity", schema=Activity)
        table.create_fts_index("text")

    # adding full text search index retrospectively
    if False:
        table = db.open_table("activity")
        table.create_fts_index("text")
