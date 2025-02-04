from __future__ import annotations

import re

from datetime import date
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional

import lancedb

from config import DB_PATH
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel
from lancedb.pydantic import Vector
from langchain.docstore.document import Document as LangchainDocument


model = get_registry().get("openai").create(name="text-embedding-ada-002")


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
    #
    time_added: datetime
    # vector: Vector(model.ndims())  #this is the vector of the Document title ... experimental

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

    def __eq__(self, other: object) -> bool:
        """Self-explanatory"""
        if not isinstance(other, Document):
            return False
        return self.location == other.location

    def __hash__(self) -> int:
        """Self-explanatory"""
        return hash(self.location)

    def as_metadata(self) -> Dict:
        """Put important fields in a dict so LanceDB Document and
        Chunk objects can easily be converted into Langchain Documents"""  # noqa
        return self.__dict__

    def is_pdf(self) -> bool:
        """Test whether the document is a PDF"""
        return self.location[-4:].lower() == ".pdf"


class Chunk(LanceModel):
    """Defines the fields which a Chunk Record contains in the LanceDB database"""

    text: str  # = model.SourceField()
    vector: Vector(model.ndims())  # = model.VectorField()
    source: Document
    order_index: Optional[
        int
    ] = None  # Only Optional because Chunks in ccid_demo_db don't have it; shouldn't be Optional in later versions

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

    @property
    def metadata(self) -> Dict:
        """Chunk metadata derived from source"""
        return self.source.as_metadata()

    @staticmethod
    def to_LangchainDocument_(text: str, metadata: Dict, enumeration_index: Optional[int] = None) -> LangchainDocument:
        """
        Convert text into a Langchain Document

        Having this as a separate method to to_LangchainDocument is useful when merging chunks
        """
        if enumeration_index:
            #   text = f"[Source ID: {enumeration_index}] {text}"
            text = f"[{enumeration_index}] {text}"
        return LangchainDocument(page_content=text, metadata=metadata)

    def to_LangchainDocument(self, **kwargs) -> LangchainDocument:
        """Convert a Chunk into a Langchain Document"""
        return self.to_LangchainDocument_(self.text, self.metadata, **kwargs)


if False:

    # to think about another time - Document-Project and Document-Unit are many-to-many relationships
    # ... which would mean a List[Project] and List[Unit] type specification in Document class
    # ... which throws an error message

    class Project(LanceModel):
        """Defines the fields which a Project nested field contains in the LanceDB database"""

        # Experimental – not sure we will ultimately need this, but it might be useful for adding
        # sophistication to retrievel methods later
        name: str

    class Unit(LanceModel):
        """Defines the fields which a Unit nested field contains in the LanceDB database"""

        # As above: experimental
        name: str


if __name__ == "__main__":

    # creata a database with a Document table and a Chunk table
    db = lancedb.connect(DB_PATH)

    # creating tables
    db.create_table(
        "document", schema=Document
    )  # this has been included experimentally and past versions but is not really needed
    # and could lead to data redundancy. See similar comment in ingest function in retrieval/db/ingest.py
    chunk_table = db.create_table("chunk", schema=Chunk)
    chunk_table.create_fts_index("text")  # this allows full text search of chunk text

    # adding full text search index retrospectively
    if False:
        table = db.open_table("chunk")
        table.create_fts_index("text")

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
