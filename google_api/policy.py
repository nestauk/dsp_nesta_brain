from datetime import datetime
from typing import List

import lancedb

from config import DB_PATH
from pydantic import BaseModel
from pydantic import Field
from retrieval.db.schema.nesta_brain import Chunk
from retrieval.db.schema.nesta_brain import Document as LanceDocument


class Policy(BaseModel):
    """A class for describing templates for Nesta policy documents."""

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

    def __repr__(self) -> str:
        """Self-explanatory"""
        format_ = "\n\tfield_id: {file_id}\n\ttitle: {title}\n\tdate_pub: {date_pub}"
        return format_.format(**{k: getattr(self, k) for k in self.__class__.dict(self) if k in format_})

    @staticmethod
    def list_as_string() -> str:
        """Get the list of policies as a string"""
        policies = get_policies()
        return "\n".join([repr(policy) for policy in policies])


def get_policies() -> List[Policy]:
    """Get all the policies in the vector DB and convert to the Policy class"""

    db = lancedb.connect(DB_PATH)

    chunk_table = db.open_table(
        "chunk"
    )  # using the chunk table instead of the document table in case the latter is deleted

    chunks = chunk_table.search().where('source.drive_type = "policy"').to_pydantic(Chunk)

    policies = {chunk.source for chunk in chunks}
    policies = [Policy(doc) for doc in policies]

    return policies


if __name__ == "__main__":

    policies = get_policies()
# print(policies)
