from __future__ import annotations

import re

from abc import abstractmethod
from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd

from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from pydantic import Field


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

    @abstractmethod
    # also a @staticmethod but can't use both decorators
    def list() -> List[BaseDriveDoc]:
        """List all the relevant drive docs"""
        pass

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

    @abstractmethod
    # Also a @property but can't use both decorators
    def prompt_template(self) -> str:
        """
        Return a string for a prompt template to use in a LangGraph node relating to the document

        Use single curly braces {list} to insert the results of the BaseDriveDoc.list() method
        Use double curly braces for fields to be filled in when the graph is invoked, e.g. {{input}}
        """
        pass

    @property
    def prompt(self) -> str:
        """
        Return a prompt to use in a LangGraph node relating to the document

        Use double curly braces for
        """
        if "{list}" in self.prompt_template:  # may be time-consuming
            template = self.prompt_template.format(list=self.list())
        else:
            template = self.prompt_template

        input_variables = re.findall(r"\{\{([^}]+)\}\}", template)

        return PromptTemplate(template=template, input_variables=input_variables)

    def to_dict(self, drop: List[str] = None, order: List[str] = None) -> OrderedDict:
        """Return fields as an ordered dictionary, for example, for conversion into a row in a dataframe"""
        if order:
            return OrderedDict({k: self.__dict__[k] for k in order})
        else:
            return OrderedDict({k: v for k, v in self.__dict__.items() if k not in (drop or [])})
