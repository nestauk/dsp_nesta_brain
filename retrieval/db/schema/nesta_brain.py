from __future__ import annotations

import re

from datetime import date
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional

import lancedb

from config import DB_PATH
from dsp_nesta_brain import logger
from lancedb.pydantic import LanceModel
from retrieval.db.schema.base import BaseChunk


class Document(LanceModel):
    """Defines the fields which a Document Record contains in the LanceDB database"""

    location: str
    title: str
    date_pub: Optional[date] = None
    # extra metadata from metadata.jsonl
    projects: Optional[List[str]] = None
    units: Optional[List[str]] = None
    rank: Optional[int] = None
    views: Optional[int] = None
    areas_of_work: Optional[List[str]] = None
    missions: Optional[List[str]] = None
    authors: Optional[List[str]] = None
    contentType: Optional[str] = None
    # other
    drive_type: Optional[str] = None
    time_added: datetime

    def __init__(self, ingestion: bool = False, **kwargs) -> None:

        if ingestion:  # this code is only needed at the time of ingestion, not when retrieving Documents from the DB
            # correcting field names
            if kwargs.get("url"):
                kwargs["location"] = kwargs.pop("url")
            if kwargs.get("publishDate"):
                kwargs["date_pub"] = kwargs.pop("publishDate")
            if kwargs.get("areasOfWork"):
                kwargs["areas_of_work"] = kwargs.pop("areasOfWork")

            # cleaning/formatting fields
            kwargs = {k: v for k, v in kwargs.items() if v != ""}  # get rid of empty strings

            if kwargs.get("areas_of_work"):  # correction of frequent problem
                kwargs["areas_of_work"] = kwargs["areas_of_work"].replace("&amp;", "and")

            for field_name in [
                "areas_of_work",
                "missions",
                "projects",
                "units",
                "authors",
            ]:  # convert string to list of strings
                if kwargs.get(field_name):
                    kwargs[field_name] = re.split(",", kwargs[field_name])

            if kwargs.get("date_pub") and type(kwargs.get("date_pub")) is str:  # convert string to date
                kwargs["date_pub"] = datetime.strptime(kwargs.get("date_pub"), "%Y-%m-%d").date()

            if False:
                # maybe for a future version – see below
                # conversion to schema class
                for field_name in ["projects", "units"]:
                    if kwargs.get(field_name):
                        class_ = globals()[field_name[0:-1].upper()]
                        kwargs[field_name] = [class_(string) for string in kwargs[field_name]]

            # additional fields
            kwargs[
                "time_added"
            ] = datetime.now()  # this does not need to be a super-accurate time, for example, to the second;
            # its purpose is to be able to filter on how recently documents were added if we want to

        super().__init__(**kwargs)

        if self.is_on_drive and not self.drive_type:
            logger.warning(f"Document {self.title}, {self.location} is on Google Drive but has no drive_type")

    def __eq__(self, other: object) -> bool:
        """Self-explanatory"""
        if not isinstance(other, Document):
            return False
        return self.location == other.location

    def __hash__(self) -> int:
        """Self-explanatory"""
        return hash(self.location)

    @property
    def is_on_drive(self) -> bool:
        """Test whether the document is on Google Drive"""
        return "drive.google.com" in self.location

    def as_metadata(self) -> Dict:
        """Put important fields in a dict so LanceDB Document and
        Chunk objects can easily be converted into Langchain Documents"""  # noqa
        return self.__dict__


class Chunk(BaseChunk):
    """Defines the fields which a Chunk Record contains in the LanceDB database"""

    source: Document

    def __init__(
        self, order_index: Optional[int] = None, **kwargs
    ) -> None:  # without explicitly including order_index as a keyword argument here
        # an error will be thrown when Chunks lacking an order_index are retrieved from the db
        super().__init__(order_index=order_index, **kwargs)

    def __eq__(self, other: object) -> bool:
        """Self-explanatory"""
        if not isinstance(other, Chunk):
            return False
        return self.source == other.source and self.text == other.text

    def __hash__(self) -> int:
        """Self-explanatory"""
        return hash(self.source.location + self.text)

    @staticmethod
    def reference_html_format() -> str:
        """Return format for references in HTML"""
        return '<a href="{url}">[{index}] {title}{pdf}</a>'

    @staticmethod
    def reference_metadata(**metadata) -> Dict:
        """Metadata useful to presentation of references"""
        metadata["url"] = metadata.get("location")
        metadata["pdf"] = " (PDF)" if metadata["location"].lower()[-4:] == ".pdf" else ""
        return metadata

    @property
    def metadata(self) -> Dict:
        """Chunk metadata derived from source"""
        return self.source.as_metadata()


if __name__ == "__main__":

    # creata a database with a Document table and a Chunk table
    db = lancedb.connect(DB_PATH)

    # creating tables
    if False:
        db.create_table("document", schema=Document)
        table = db.create_table("chunk", schema=Chunk)
        table.create_fts_index("text")

    # adding full text search index retrospectively
    if False:
        table = db.open_table("chunk")
        table.create_fts_index("text")

    # searching for records
    if True:
        document_table = db.open_table("document")
        chunk_table = db.open_table("chunk")

        #   docs = document_table.search().where('title LIKE "%Birthing Parent%"').limit(10).to_pydantic(Document)
        #  print(docs)
        chunks = (
            chunk_table.search()
            .where('source.location LIKE "https://drive.google.com/file/d%"')
            .limit(100)
            .to_pydantic(Chunk)
        )
    #   print(chunks)

    # fixing a cock up
    if False:
        # import pandas as pd
        copy_from_path = "retrieval/db/full_site_demo_db_first_attempt"
        copy_from_db = lancedb.connect(copy_from_path)
        document_table = copy_from_db.open_table("document")
        chunk_table = copy_from_db.open_table("chunk")

        docs = document_table.search().where('NOT location LIKE "%.pdf"').limit(1000000).to_pydantic(Document)
        chunks = chunk_table.search().where('NOT source.location LIKE "%.pdf"').limit(1000000).to_pydantic(Chunk)

        new_doc_table = db.create_table("document", schema=Document)
        new_doc_table.add(docs)
        new_chunk_table = db.create_table("chunk", schema=Chunk)
        new_chunk_table.add(chunks)
        new_chunk_table.create_fts_index("text")
