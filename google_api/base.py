from __future__ import annotations

import re

from abc import abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd

from dsp_nesta_brain import logger
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from llm.llm import default_llm as llm
from pydantic import BaseModel
from pydantic import Field


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from lgraph.graph import State


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

    @abstractmethod
    # also a @staticmethod but can't use both decorators
    def list() -> List[BaseDriveDoc]:
        """List all the relevant drive docs"""
        pass

    @classmethod
    def decide_whether_needs_document(cls, state: State) -> State:
        """Decide whether a drive document is needed from an input message and return the file IDs if so"""

        chain = RunnablePassthrough.assign(input=(lambda x: x["messages"][-1])) | cls.prompt() | llm

        message = chain.invoke(state)

        if message.content != "NULL":

            file_ids = message.content
            file_ids = [file_id.strip() for file_id in file_ids.split(",")]
            file_ids_are_right_format = all(re.search(r"^[A-Za-z0-9\-_]+$", file_id) for file_id in file_ids)

            if file_ids_are_right_format:
                logger.info(f"{cls.__name__} IDs identified: {file_ids}")
                state["intermediate_outputs"]["file_ids"] = file_ids

            else:
                logger.warning(
                    f"{cls.__name__} IDs did not seem to be in the correct format. Message content: {message.content}"
                )

        return state

    @abstractmethod
    # Also a @classmethod but can't use both decorators
    def prompt_template(cls) -> str:
        """
        Return a string for a prompt template to use in a LangGraph node relating to the document

        Use single curly braces {list} to insert the results of the BaseDriveDoc.list() method
        Use double curly braces for fields to be filled in when the graph is invoked, e.g. {{input}}
        """
        pass

    @classmethod
    def prompt(cls) -> PromptTemplate:
        """
        Return a prompt to use in a LangGraph node relating to the document

        Use double curly braces for
        """

        template = cls.prompt_template()

        if "{list}" in template:  # may be time-consuming, so only do if needed
            template = template.format(list=cls.list(as_string=True))

        input_variables = re.findall(r"\{\{([^}]+)\}\}", template)

        return PromptTemplate(template=template, input_variables=input_variables)

    @classmethod
    def sub_graph(cls) -> CompiledStateGraph:
        """Return the subgraph for the Policy class"""

        builder = StateGraph(State)

        builder.add_node("decide_whether_needs_document", cls.decide_whether_needs_document)
        builder.add_edge(START, "decide_whether_needs_document")
        builder.add_edge("decide_whether_needs_document", END)

        graph = builder.compile()
        return graph

    def to_dict(self, drop: List[str] = None, order: List[str] = None) -> OrderedDict:
        """Return fields as an ordered dictionary, for example, for conversion into a row in a dataframe"""
        if order:
            return OrderedDict({k: self.__dict__[k] for k in order})
        else:
            return OrderedDict({k: v for k, v in self.__dict__.items() if k not in (drop or [])})
