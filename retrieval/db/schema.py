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
    contentType: Optional[List[str]] = None
    #
    time_added: datetime
    # vector: Vector(model.ndims())  #this is the vector of the Document title ... experimental

    def __init__(self, **kwargs) -> None:

        # correcting field names
        if kwargs.get("url"):
            kwargs["location"] = kwargs.pop("url")
        if kwargs.get("publishDate"):
            kwargs["date_pub"] = kwargs.pop("publishDate")

        # cleaning/formatting fields
        if kwargs.get("areasOfWork"):
            kwargs["areasOfWork"] = kwargs["areasOfWork"].replace("&amp;", "and")
        for field_name in ["areasOfWork", "missions", "projects", "units"]:
            if kwargs.get(field_name):
                kwargs[field_name] = re.split(",", kwargs[field_name])

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
        return {attr: getattr(self, attr) for attr in ["location", "title", "date_pub"]}

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
    ]  # Only Optional because Chunks already in the db won't have it; shouldn't be Optional in later versions

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

    @property
    def metadata(self) -> Dict:
        """Chunk metadata derived from source"""
        return self.source.as_metadata()

    def to_LangchainDocument(self) -> LangchainDocument:
        """Convert a Chunk into a Langchain Document"""
        return LangchainDocument(page_content=self.text, metadata=self.metadata)


if False:

    # to think about another time - Document-Project and Document-Unit are many-to-many relationships
    # which would mean a List[Project] and List[Unit] type specification in Document class
    # which is throwing an error message ...

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

    table = db.create_table("document", schema=Document)
    table = db.create_table("chunk", schema=Chunk)
