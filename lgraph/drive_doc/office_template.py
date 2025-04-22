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

# from lgraph.graph import call_default_chain
from lgraph.research_agent.research_agent import AgentState as State

# from lgraph.research_agent.research_agent import revise as research_agent_revise
from llm.llm import default_llm as llm
from llm.message import InterimAIMessage
from llm.prompt import qa_system_prompt


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
            "check_template": check_template,
            "fetch_template": fetch_template,
            "apply_template": apply_template,  # test_apply_template
            "check_revise_or_upload": check_revise_or_upload,
            #  "revise": research_agent_revise,
            "upload_output": upload_output,
            #    "call_default_chain": call_default_chain,
            "conclude": conclude,
        }

        default_nodes.update(nodes)
        nodes = default_nodes

        for node_name, node_func in nodes.items():
            builder.add_node(node_name, node_func)

        builder.add_edge(START, "decide_whether_needs_template")
        builder.add_conditional_edges("decide_whether_needs_template", template_router)
        builder.add_conditional_edges("check_template", check_template_router)
        builder.add_edge("fetch_template", "apply_template")
        builder.add_edge("apply_template", "check_revise_or_upload")
        #  builder.add_edge("revise", "conclude")
        builder.add_conditional_edges("check_revise_or_upload", revise_or_upload_router)
        builder.add_edge("upload_output", "conclude")
        #  builder.add_edge("call_default_chain", "conclude")
        builder.add_edge("conclude", END)

        if add_checkpoints:
            logger.info("Adding checkpoints to OfficeTemplate graph")
            interrupt_before = ["check_template", "check_revise_or_upload"]
            memory = MemorySaver()
            return builder.compile(interrupt_before=interrupt_before, checkpointer=memory), interrupt_before

        logger.info("Didn't add checkpoints to the graph")
        graph = builder.compile()
        return graph, None


# -----nodes


# def test_apply_template(state: State) -> State:
#   """Set draft with a test message"""
#  state["draft"] = "This is a test draft"
# return state


def check_template(state: State) -> State:
    """
    Placeholder: Check whether the template selected by decide_whether_needs_template is the right one

    This doesn't do anything at the moment because the actual checking happens in the app.

    This is needed for the graph structure to be clearer, but it's not used in practice.
    """  # noqa

    return state


def check_revise_or_upload(state: State) -> State:
    """
    Placeholder: Check whether the upload should proceed

    This doesn't do much at the moment because the actual checking happens in the app.

    This is needed for the graph structure to be clearer, but it's not used in practice.
    """  # noqa
    #    print("In check_revise_or_upload")
    return state


def fetch_template(state: State) -> State:
    """Retrieve an office document template from Google Drive to help write a document"""

    file_ids_dict = state["intermediate_outputs"].get("file_ids")
    if file_ids_dict:

        _, file_ids = list(file_ids_dict.items())[
            -1
        ]  # file ids returned from the last iteration of decide_whether_needs_document (there will usually only be one)
        # see BaseDriveDoc.decide_whether_needs_document for a description of this dict's structure
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


def conclude(state: State) -> State:
    """
    Do nothing and return the state.
    This is only needed because it is unclear how to use END in conditional edges
    """  # noqa
    return state


# -----conditional edges


def template_router(
    state: State,
) -> Literal["conclude", "decide_whether_needs_template", "check_template"]:
    """Go the appropriate node, depending on whether an office template is needed"""

    if state.get("router_override"):
        return state["router_override"]  # the routing wasn't behaving as desired when users reject the template
        # this was added to force the desired behaviour

    file_ids_dict = state.get("intermediate_outputs", {}).get("file_ids")

    if file_ids_dict:

        call_no, file_ids = list(file_ids_dict.items())[-1]

        if file_ids and file_ids[0] in OfficeTemplate.file_ids():
            return "check_template"

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

    else:
        return "check_template"  # go to the checkpoint before check_template and tell the user no template was found


def check_template_router(state: State) -> Literal["fetch_template", "conclude"]:
    """Go the appropriate node, depending on whether the template is the right one"""

    if state["intermediate_outputs"].get("file_ids"):
        return "fetch_template"
    else:
        return "conclude"


def revise_or_upload_router(state: State) -> Literal["upload_output", "conclude"]:  # ,"revise"]:
    # """Go the appropriate node, depending on whether the output should be revised or uploaded to Google Drive"""
    """Go the appropriate node, depending on whether the output should be uploaded to Google Drive"""

    if state.get("upload_confirmed"):
        return "upload_output"

    # elif re.search("^HUMAN CRITIQUE:.+", state.get("critique", "")):
    #    return "revise"

    else:
        return "conclude"
