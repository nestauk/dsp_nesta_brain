from __future__ import annotations

import math
import re

from datetime import date
from datetime import datetime
from typing import Dict
from typing import List
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
    recipient_country_code: Optional[str] = None
    sector_narrative: Optional[str] = None
    activity_status_code: int
    activity_date_iso_date: List[date]
    activity_date_type: List[int]
    activity_date_narrative: Optional[str]
    document_link_url: Optional[str] = None
    document_link_title_narrative: Optional[str] = None
    document_link_description_narrative: Optional[str]
    min_date: date
    max_date: date
    min_year: int
    max_year: int
    activity_status: str
    text: str

    time_added: datetime  # not Optional for Activity

    def __init__(self, ingestion: bool = False, **kwargs) -> None:

        for k, v in kwargs.items():

            if (type(v) is float and math.isnan(v)) or (type(v) is str and v.lower() == "nan"):
                kwargs[k] = None

            else:

                field_info = Activity.model_fields.get(k)

                if field_info:

                    type_ = field_info.annotation

                    if "typing.List" in str(type_) and type(v) is str:
                        list_elements = re.split(",", v)
                        if "[int]" in str(type_):
                            list_elements = [int(ele) for ele in list_elements]
                        elif "[date]" in str(type_):
                            list_elements = [datetime.fromisoformat(ele).date() for ele in list_elements]
                        v = list_elements

                    elif type_ is date and type(v) is str:
                        v = datetime.fromisoformat(v).date()

                    elif type_ is int and type(v) is str:
                        v = int(v)

                    kwargs[k] = v

        if ingestion:
            kwargs["time_added"] = datetime.now()

        super().__init__(**kwargs)

    def __eq__(self, other: object) -> bool:
        """Self-explanatory"""
        if not isinstance(other, Activity):
            return False
        return self.iati_identifier == other.iati_identifier

    def __hash__(self) -> int:
        """Self-explanatory"""
        return hash(self.iati_identifier)

    def __repr__(self) -> str:
        """Self-explanatory"""
        return self.iati_identifier

    @property
    def description_narrative(self) -> str:
        """Return description_narrative field dynamically rather than duplicating data"""
        return self.text.replace(self.title, "", 1).strip()

    @property
    def metadata(self) -> Dict:
        """Activity metadata"""
        metadata = {k: v for k, v in self.__dict__.items() if k not in ["time_added", "text"]}
        return metadata


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
