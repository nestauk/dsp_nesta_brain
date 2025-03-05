from __future__ import annotations

import importlib

from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING
from typing import List

import lancedb

from config import DB_PATH
from google_api.base import BaseDriveDoc
from pydantic import Field
from retrieval.db.schema.nesta_brain import Chunk
from retrieval.db.schema.nesta_brain import Document as LanceDocument


if TYPE_CHECKING:
    from lgraph.graph import State


class Policy(BaseDriveDoc):
    """A class for describing Nesta policy documents."""

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

    @staticmethod
    def list(**kwargs) -> List[Policy]:
        """
        Get all the policies in the vector DB and convert to the Policy class

        The policy documents should all have been ingested into the database; deriving a list from
        the database is faster than reading the documents from the Google Drive API.
        """

        db = lancedb.connect(DB_PATH)

        chunk_table = db.open_table(
            "chunk"
        )  # using the chunk table instead of the document table in case the latter is deleted

        chunks = chunk_table.search().where('source.drive_type = "policy"').limit(-1).to_pydantic(Chunk)

        policies = [Policy(chunk.source) for chunk in chunks]

        if kwargs:
            return Policy.format_list(policies, **kwargs)

        else:
            return policies

    @classmethod
    def decide_whether_needs_document(cls, state: State) -> State:
        """Decide whether a policy document is needed to answer a query base on a state and amend the state accordingly if so"""

        state = super(cls, Policy).decide_whether_needs_document(state)
        file_ids = state["intermediate_outputs"].get("file_ids")

        if file_ids:

            filter_condition = "(" + " or ".join([f'source.location LIKE "%{file_id}"' for file_id in file_ids]) + ")"
            state["use_hybrid_search"] = False
            # filter_condition = f'(source.drive_type == "policy" or source.location LIKE "%{file_id}")'
            state = importlib.import_module("lgraph.graph").append_filter_condition(
                state, filter_condition
            )  # avoiding circular import

        return state

    @classmethod
    def prompt_template(cls) -> str:
        """Return a string for a prompt template to use in a LangGraph node relating to the document"""

        return """
            You are a helpful assistant and an expert on the internal administration and organisational policies of the innovation agency Nesta.

            Your role is to help staff with their queries about organisational policies. Topics include annual leave, expenses, safeguarding, and more.

            Look at the following list of policy documents available to help you answer queries:

            List:
            {list}

            Look at the query below and decide whether one or more the policies in the list is needed to answer it.

            If so, select the appropriate policies. As your response, give only the UIDs of the policy you have selected separated by commas.

            Otherwise, respond "NULL".

            Query:
            {{input}}
            """  # noqa

    def to_dict(self) -> OrderedDict:
        """Return fields as an ordered dictionary, for example, for conversion into a row in a dataframe"""

        dict_ = self.__dict__
        dict_["link"] = f"https://drive.google.com/file/d/{dict_['file_id']}"
        return OrderedDict({k: dict_[k] for k in ["title", "date_pub", "link"]})
