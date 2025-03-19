from __future__ import annotations

import importlib
import re

from collections import OrderedDict
from typing import TYPE_CHECKING

from dsp_nesta_brain import logger
from google_api.drive_doc.base import BaseDriveDoc
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from llm.llm import default_llm as llm
from llm.parser import force_enum_parser_multi


if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from langgraph.graph.state import CompiledStateGraph
    from lgraph.graph import State

pause = input


class BaseDriveDoc(BaseDriveDoc):
    """A class for describing templates for documents on Google Drive (note: need not be a Google Doc)."""

    @classmethod
    def decide_whether_needs_document(cls, state: State) -> State:
        """Decide whether a drive document is needed from an input message and return the file IDs if so"""

        prompt = cls.prompt()

        chain = (
            RunnablePassthrough.assign(input=(lambda x: x["messages"][-1])) | prompt | llm | cls.enum_output_parser()
        )

        file_ids_as_enums = chain.invoke(state)

        if file_ids_as_enums:
            file_ids = [enum.value for enum in file_ids_as_enums]
            logger.info(f"{cls.__name__} ID(s) identified: {file_ids}")
            state.setdefault("intermediate_outputs", {})
            call_no = len(state["intermediate_outputs"].get("file_ids", OrderedDict({}))) + 1
            state["intermediate_outputs"].setdefault("file_ids", {})[call_no] = file_ids

        return state

    @classmethod
    def enum_output_parser(cls) -> Runnable:
        """Return a parser which forces LLM output into a list of enums representing file IDs"""
        enum = cls.file_ids(as_enum=True)
        return force_enum_parser_multi(enum)

    # @classmethod
    # def prompt_template(cls) -> str:
    #   """
    #  Return a string for a prompt template to use in a LangGraph node relating to the document
    ##
    #      Use single curly braces {list} to insert the results of the BaseDriveDoc.list() method
    #     Use double curly braces for fields to be filled in when the graph is invoked, e.g. {{input}}
    #    """
    #   pass

    @classmethod
    def prompt_template(cls) -> str:
        """Return a string for a prompt template to use in a LangGraph node relating to the document"""

        return """
            You are a helpful assistant and an expert in the internal administration, personnel, projects and organisational policies of the innovation agency Nesta.

            Your role is to help staff with their queries about organisational policies. Topics include annual leave, expenses, safeguarding, and more.

            You can also decide whether they need a document to help them write a proposal, project update, etc., and provide them with the relevant template.

            Look at the query below and decide whether the staff member needs a policy document or office template to help answer it.

            If so, select the appropriate policy document or office template from the POLICY list or the TEMPLATE list. As your response, give only the UIDs of the documents you have selected separated by commas.

            Otherwise, respond "NULL".

            Query:
            {{input}}

            POLICY list:
            {policy_list}

            TEMPLATE list:
            {office_template_list}
            """  # noqa

    @classmethod
    def prompt(cls, combined: bool = False) -> PromptTemplate:
        """Return a prompt to use in a LangGraph node relating to the document"""

        template = cls.prompt_template()

        doc_list, policy_list, office_template_list = None, None, None

        if "{list}" in template:  # may be time-consuming, so only do if needed
            doc_list = cls.list(as_string=True)

        if "{policy_list}" in template:
            policy_list = importlib.import_module("lgraph.drive_doc.policy").Policy.list(as_string=True)

        if "{office_template_list}" in template:  # time-consuming, so only do if needed
            office_template_list = importlib.import_module("lgraph.drive_doc.office_template").OfficeTemplate.list(
                as_string=True
            )

        template = template.format(list=doc_list, policy_list=policy_list, office_template_list=office_template_list)

        input_variables = re.findall(r"\{\{([^}]+)\}\}", template)

        return PromptTemplate(template=template, input_variables=input_variables)

    @classmethod
    def sub_graph(cls) -> CompiledStateGraph:
        """Return an appropriate subgraph"""

        builder = StateGraph(State)

        builder.add_node("decide_whether_needs_document", cls.decide_whether_needs_document)
        builder.add_edge(START, "decide_whether_needs_document")
        builder.add_edge("decide_whether_needs_document", END)

        graph = builder.compile()
        return graph
