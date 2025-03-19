from __future__ import annotations

import re

from collections import OrderedDict
from enum import Enum
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd

from pydantic import BaseModel
from pydantic import Field


if TYPE_CHECKING:
    from enum import EnumType


class BaseDriveDoc(BaseModel):
    """A class for describing templates for documents on Google Drive (note: need not be a Google Doc)."""

    file_id: str = (
        Field(
            ...,
            description="A unique identifier",
        ),
    )
    title: str = Field(
        ...,
        description="The title of the document",
    )

    def __init__(self, google_api_data: Optional[Dict] = None, **kwargs) -> None:

        if google_api_data:
            kwargs["file_id"] = google_api_data["id"]
            if "title" not in kwargs:
                kwargs["title"] = google_api_data["name"]

        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Self-explanatory"""
        return "\n\t" + "\n\t".join([f"{k}: {v}" for k, v in self.__dict__.items()])

    @staticmethod
    def format_list(
        base_docs: List[BaseDriveDoc], as_string: bool = False, to_csv: bool = False
    ) -> Union[str, pd.DataFrame]:
        """Get all the policies in the vector DB and convert to the Policy class"""

        if to_csv:
            df = pd.DataFrame.from_records([doc.to_dict() for doc in base_docs])
            df.to_csv("retrieval/db/ingest/policies.csv", index=False)
            return df

        elif as_string:
            return "\n".join([repr(doc) for doc in base_docs])

    # @abstractmethod but can't use both decorators
    @staticmethod
    def list() -> List[BaseDriveDoc]:
        """List all the relevant drive docs"""
        pass

    @classmethod
    def list_as_dict(cls) -> Dict[str, BaseDriveDoc]:
        """Return a dictionary of all the relevant drive docs"""
        return {doc.file_id: doc for doc in cls.list()}

    @classmethod
    def description(cls) -> str:
        """Return a description of the document class"""
        pascal_case_components = re.split("([A-Z][a-z]+)", cls.__name__)
        return " ".join([c for c in pascal_case_components if c])

    @classmethod
    def file_ids(cls, as_enum: bool = False, as_dict: bool = False) -> Union[List[str], EnumType]:
        """Return a list of file IDs, or an EnumType representing file IDs"""
        file_ids = [doc.file_id for doc in cls.list()]
        if as_enum:
            return Enum(cls.__name__ + "Enum", {file_id: file_id for file_id in file_ids})
        else:
            return file_ids

    def to_dict(self, drop: List[str] = None, order: List[str] = None) -> OrderedDict:
        """Return fields as an ordered dictionary, for example, for conversion into a row in a dataframe"""
        if order:
            return OrderedDict({k: self.__dict__[k] for k in order})
        else:
            return OrderedDict({k: v for k, v in self.__dict__.items() if k not in (drop or [])})
