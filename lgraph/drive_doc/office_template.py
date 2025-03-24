from __future__ import annotations

import importlib
import re

from typing import TYPE_CHECKING
from typing import Literal
from typing import Optional

import streamlit as st

from dsp_nesta_brain import logger
from google_api.drive import create_document_in_folder_from_markdown
from google_api.drive import create_document_in_folder_from_string
from google_api.drive import get_document
from google_api.drive_doc.office_template import OfficeTemplate
from googleapiclient.errors import HttpError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableParallel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from lgraph.drive_doc.base import BaseDriveDoc
from lgraph.graph import call_default_chain
from lgraph.research_agent.research_agent import AgentState as State
from llm.llm import default_llm as llm
from llm.message import InterimAIMessage
from llm.prompt import qa_system_prompt
from utils import yesno


# THIS IS AN ADAPTED VERSION OF PREVIOUS docgen branch CODE AND HASN'T BEEN TESTED YET


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


MAX_ROUTER_RETRIES = 3


apply_template_appendix = """

    Finally, apply the given template to structure your answer. Give only the final document in Markdown as your answer.

    TEMPLATE:
    {template}
"""


class OfficeTemplate(OfficeTemplate, BaseDriveDoc):
    """A class for describing templates for proposals, project updates, etc."""

    @classmethod
    def prompt_template(cls) -> str:
        """Return a string for a prompt template to use in a LangGraph node relating to the document"""

        return """
            You are a helpful assistant and an expert in the internal administration, personnel and projects of the innovation agency Nesta.

            Look at the following list of office document templates available to you to help staff write documents: {list}

            Look at the request below and decide whether the one of the document templates in the list is needed to fulfil it.

            If so, select the appropriate template. As your response, give only the file_id of the template you have selected.

            Otherwise, respond "NULL".

            Request:
            {{input}}
            """  # noqa

    @classmethod
    def sub_graph(
        cls, builder: Optional[StateGraph] = None, add_checkpoints: bool = False, **nodes
    ) -> CompiledStateGraph:
        """Return the subgraph for the OfficeTemplate class"""

        builder = builder or StateGraph(State)

        default_nodes = {
            "decide_whether_needs_template": cls.decide_whether_needs_document,
            #            "check_template": check_template,
            "fetch_template": fetch_template,
            "apply_template": apply_template,
            "upload_output": upload_output,
            "call_default_chain": call_default_chain,
        }

        default_nodes.update(nodes)
        nodes = default_nodes

        for node_name, node_func in nodes.items():
            builder.add_node(node_name, node_func)

        builder.add_edge(START, "decide_whether_needs_template")
        builder.add_conditional_edges("decide_whether_needs_template", template_router)
        #   builder.add_edge("check_template", "fetch_template")
        builder.add_edge("fetch_template", "apply_template")
        builder.add_edge("apply_template", "upload_output")
        builder.add_edge("upload_output", END)
        builder.add_edge("call_default_chain", END)

        if add_checkpoints:
            memory = MemorySaver()
            return builder.compile(interrupt_before=["fetch_template"], checkpointer=memory)

        graph = builder.compile()
        return graph


# -----nodes


# def check_template(state: State) -> State:
#   """Check whether the template selected by decide_whether_needs_template is the right one"""

#  file_ids_dict = state["intermediate_outputs"].get("file_ids")
# if file_ids_dict:

#    _, file_ids = list(file_ids_dict.items())[-1]
#   file_id = file_ids[0]  # there should only be one

#  st.toast("Look at command line", icon="👀")
# input(
#    f'This is temporary and will be replaced with a checkpoint in the graph where the user answers this question via the UI\n.I think I need "{OfficeTemplate.list_as_dict()[file_id].title}" from Google Drive. Is this correct? Press any key to proceed.'  # noqa

# )
# return state


def fetch_template(state: State) -> State:
    """Retrieve an office document template from Google Drive to help write a document"""

    file_ids_dict = state["intermediate_outputs"].get("file_ids")
    if file_ids_dict:

        _, file_ids = list(file_ids_dict.items())[-1]
        file_id = file_ids[0]  # there should only be one, but test this

        st.toast("Fetching template from Google Drive", icon="💡")

        try:
            google_doc = get_document(file_id, as_google_doc=True)
            state["intermediate_outputs"]["template"] = google_doc
            return state

        except HttpError as http_error:
            logger.error(f'Error finding Google Doc with ID "{file_id}": {http_error}')


def apply_template(state: State) -> State:
    """
    Apply a template retrieved from Drive to format the input request

    N.B. this is not used if the research agent subgraph is used instead
    """

    google_doc_template = state["intermediate_outputs"]["template"]

    prompt_template = qa_system_prompt + apply_template_appendix
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder("messages"),
        ]
    )
    state["messages"] = [msg for msg in state["messages"] if not isinstance(msg, InterimAIMessage)]

    chain = (
        RunnableParallel(
            template=(lambda: google_doc_template.text),
            context=(lambda x: x["context"]),
            messages=(lambda x: x["messages"]),
        )
        | prompt
        | llm
    )

    rag_chain = importlib.import_module("llm.chain").history_aware_rag_chain(chain)

    logger.info(f'Applying template "{google_doc_template.title}" to fulfil the request')
    response = rag_chain.invoke(state)

    state["messages"].append(response["answer"])

    return state


def upload_output(state: State) -> State:
    """
    Upload the content of the last message to Google Drive.

    Ideally the output should be in Markdown format
    """

    file_name = "research_agent_output.md"
    output = state["messages"][-1].content
    markdown_title_match = re.search("[*#]", output)

    if not markdown_title_match:
        file_name = file_name.replace(".md", ".txt")

    st.toast(f"Uploading the output to {file_name} in Google Drive", icon="📤")

    if markdown_title_match:
        markdown = output[markdown_title_match.span()[0] :]
        logger.info(f"Research agent output being uploaded to file {file_name}")
        create_document_in_folder_from_markdown(markdown, file_name=file_name)
    else:
        logger.warning(
            f"Markdown not detected in LLM output – research agent output being uploaded to a text file {file_name}"
        )
        create_document_in_folder_from_string(output, file_name=file_name)

    return state


# -----conditional edges


def template_router(
    state: State,
) -> Literal["check_template", "call_default_chain", "decide_whether_needs_template"]:
    """Go the appropriate node, depending on whether an office template is needed"""

    file_ids_dict = state["intermediate_outputs"].get("file_ids")

    if file_ids_dict:
        call_no, file_ids = list(file_ids_dict.items())[-1]

        if file_ids and file_ids[0] in OfficeTemplate.file_ids():
            return "fetch_template"

        else:
            invalid_template_message = "decide_whether_needs_template node returned an invalid template UID"
            # this should no longer be necessary now that the LLM output is forced to be a template UID via
            # force_enum_parser_multi
            # maybe remove this?

            if call_no < MAX_ROUTER_RETRIES:
                logger.info(invalid_template_message + " – retrying")
                return "decide_whether_needs_template"

            else:
                logger.info(
                    invalid_template_message
                    + " - however, the maximum number of retries for decide_whether_needs_template node have already been met. Proceeding without using template"  # noqa
                )

    return "call_default_chain"


def check_template_router(state: State) -> Literal["fetch_template", "wash_up"]:
    """Check whether the template selected by decide_whether_needs_template is the right one"""

    if True:
        return "fetch_template"
    else:
        return "wash_up"


def upload_router(state: State) -> Literal["upload_output", "wash_up"]:
    """Go the appropriate node, depending on whether the output should be uploaded to Google Drive"""

    if yesno("Upload the output?"):  # temporary decision process to check it works
        # – obviously users ultimately won't be interacting with this via the command line
        return "upload_output"

    return "wash_up"
