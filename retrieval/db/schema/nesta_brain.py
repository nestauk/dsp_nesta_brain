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

        if ingestion and self.is_on_drive and not self.drive_type:
            logger.warning(
                f"Document {self.title}, {self.location} is on Google Drive but has no drive_type. Consider specifying drive_type via ingestion settings."  # noqa
            )  # noqa

    def __eq__(self, other: object) -> bool:
        """Self-explanatory"""
        if not isinstance(other, Document):
            return False
        return self.location == other.location

    def __hash__(self) -> int:
        """Self-explanatory"""
        return hash(self.location)

    @property
    def file_id(self) -> str:
        """Extract the file ID from Drive location"""
        if self.is_on_drive:
            return self.location.replace("https://drive.google.com/file/d/", "")

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
    def reference_html_format(add_date: bool = False) -> str:
        """Return format for references in HTML"""
        format = '<a href="{url}">[{index}] {title}{pdf}'
        if add_date:
            format += " {date_pub}"
        format += "</a>"
        return format

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


class MissionProject(BaseChunk):
    """Represents a record in data/Mission Project List.csv"""

    mission: Optional[str] = None
    name: str
    code: Optional[str] = None
    lifecycle_stage: Optional[str] = None
    area_of_focus: Optional[str] = None
    intermediate_goal: Optional[str] = None
    geography: Optional[str] = None

    time_added: datetime  # not Optional for MissionProject

    def __init__(self, ingestion: bool = False, **kwargs) -> None:

        #  print({k:v for k,v in kwargs.items() if k != 'vector'})

        if ingestion:

            for k, v in kwargs.items():

                if (type(v) is float and math.isnan(v)) or str(v).lower() in ["nan", "", " "]:
                    kwargs[k] = None

                elif k == "Area of Focus":
                    v = re.sub(r"[0-9]+\.", "", v)

                if type(v) is str:
                    v = v.strip()

            super().__init__(
                mission=kwargs.get("Team"),
                name=kwargs.get("Project Name (Asana)"),
                code=kwargs.get("Project Code (Nesta)"),
                lifecycle_stage=kwargs.get("Lifecycle Stage"),
                area_of_focus=kwargs.get("Area of Focus"),
                time_added=datetime.now(),
                **kwargs,
            )  # I tried doing this a clever way but kept getting error messages relating to changing the size/keys of kwargs

        else:
            super().__init__(**kwargs)

    def __eq__(self, other: object) -> bool:
        """Self-explanatory"""
        if not isinstance(other, MissionProject):
            return False
        return self.code == other.code and self.text == other.text

    def __hash__(self) -> int:
        """Define the hash value"""
        return hash((self.code or "") + self.name)

    @staticmethod
    def reference_html_format() -> str:
        """Return format for references in HTML"""
        return "[{index}] Project {code}: {name} ({lifecycle_stage})</a>"

    @staticmethod
    def reference_metadata(**metadata) -> Dict:
        """Metadata useful to presentation of references"""
        if not metadata.get("code"):
            metadata["code"] = "(unknown code)"
        return metadata

    @property
    def metadata(self) -> Dict:
        """MissionProject metadata"""  # noqa
        metadata = {k: v for k, v in self.__dict__.items() if k not in ["time_added", "text", "vector"]}
        return metadata

    @property
    def research_question(self) -> str:
        """Return research_question field dynamically rather than duplicating data"""
        return self.text.replace(self.name, "", 1).strip()


table_name_to_schema_class_map = {"chunk": Chunk, "mission_project": MissionProject}

if __name__ == "__main__":

    # creata a database with a Document table and a Chunk table
    db = lancedb.connect(DB_PATH)

    # creating tables
    if True:
        #  db.create_table("document", schema=Document)
        table = db.create_table("mission_project", schema=MissionProject)
        table.create_fts_index("text")

    # dropping tables
    if False:
        db.drop_table("mission_project")

    # adding full text search index retrospectively
    if False:
        table = db.open_table("chunk")
        table.create_fts_index("text")

    # searching for records
    if False:
        document_table = db.open_table("document")
        chunk_table = db.open_table("chunk")

        #   docs = document_table.search().where('title LIKE "%Birthing Parent%"').limit(10).to_pydantic(Document)
        #  print(docs)
        # results = chunk_table.search().where('source.location NOT LIKE "https://%"').limit(100).to_pydantic(Chunk)
        results = chunk_table.search("climate change").limit(100).select(["text"]).to_list()
        logger.info(len(results))

    # adding columns
    if False:
        table = db.open_table("document")
        table.add_columns({"drive_type": "cast(NULL as string)"})

    # deleting records
    if False:

        input("You are about to delete some records. Press any key to continue.")
        document_table = db.open_table("document")
        chunk_table = db.open_table("chunk")

        #  document_table.delete('location LIKE "https://drive.google.com/file/d%"')
        chunk_table.delete('source.location = "1NeuLG4DAHg-gd_iwAWCWKq80_iVUmXMp"')

    # updating records
    if False:

        document_table = db.open_table("document")
        chunk_table = db.open_table("chunk")

        bad_title = "Sickness Absence Policy - Update July 2022"
        good_title = "Sickness Absence Policy"
        document_table.update(where=f'title LIKE "%{bad_title}%"', values={"title": good_title})
        results = document_table.search().where(f'title LIKE "%{good_title}%"').limit(1).to_pydantic(Document)
        updated_document = results[0]

        bad_chunks = chunk_table.search().where(f'source.title LIKE "%{bad_title}%"').limit(1).to_pydantic(Chunk)
        chunk_table.delete(f'source.title LIKE "%{bad_title}%"')
        for bad_chunk in bad_chunks:
            good_chunk = bad_chunk
            good_chunk.source = updated_document
            chunk_table.add([good_chunk])

    # copying tables from one db to another
    if False:

        copy_from_path = "retrieval/db/full_site_demo_db_with_pdfs"
        copy_from_db = lancedb.connect(copy_from_path)
        copy_from_document_table = copy_from_db.open_table("document")
        copy_from_chunk_table = copy_from_db.open_table("chunk")

        copy_to_document_table = db.open_table("document")
        copy_to_chunk_table = db.open_table("chunk")

        # docs = copy_from_document_table.search().to_pydantic(Document)
        # copy_to_document_table.add(docs)

        chunks_df = (
            copy_from_chunk_table.search().limit(-1).to_pandas()
        )  # too many records to convert to pydantic straight away

        batch_size = 10000
        for i in range(0, chunks_df.shape[0], batch_size):
            rows = chunks_df.iloc[i : i + batch_size]
            chunks = [Chunk(**row.to_dict()) for _, row in rows.iterrows()]
            copy_to_chunk_table.add(chunks)

    if False:
        copy_from_path = "retrieval/db/full_site_demo_db_with_pdfs"
        copy_from_db = lancedb.connect(copy_from_path)
        copy_from_chunk_table = copy_from_db.open_table("chunk")
        chunks = copy_from_chunk_table.search().limit(None).to_pandas()  # .to_pydantic(Chunk)
    #  print(chunks.shape)
    #  copy_to_chunk_table.add(chunks)
