from datetime import date
from datetime import datetime
from typing import Dict
from typing import Optional

import lancedb

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
    time_added: datetime
    # vector: Vector(model.ndims())  #this is the vector of the Document title ... experimental

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


class Chunk(LanceModel):
    """Defines the fields which a Chunk Record contains in the LanceDB database"""

    text: str  # = model.SourceField()
    vector: Vector(model.ndims())  # = model.VectorField()
    source: Document

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


if __name__ == "__main__":

    # creata a database with a Document table and a Chunk table
    db = lancedb.connect("retrieval/db/ccid_demo_db")

    table = db.create_table("document", schema=Document)
    table = db.create_table("chunk", schema=Chunk)
