from __future__ import annotations

import logging
import os

from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING
from typing import List
from typing import Optional

import markdown
import streamlit as st

from config import DEBUG_MODE
from dotenv import load_dotenv
from front_end.sidebar import sidebar
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from lgraph.drive_doc.office_template import OfficeTemplate
from lgraph.research_agent.research_agent import create_graph
from lgraph.research_agent.research_agent import revise as research_agent_revise
from llm.message import CustomAIMessage


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from lgraph.research_agent.research_agent import AgentState as State
    from streamlit.delta_generator import DeltaGenerator

pause = input

stream_nodes = []  # temporary, to please Flake8

CURRENT_YEAR = datetime.now().year
PREVIEW_CONTAINER_HEIGHT = 500


langfuse = Langfuse()

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    user_id=os.getenv("LANGFUSE_USER_ID"),
)

config = {"configurable": {"thread_id": "1"}}


WIDGET_SPEC = OrderedDict(
    {
        "knowledge_source": {
            "default": "Nesta Vector DB",
            "filter_condition_format": None,
            "options": ("LLM internal knowledge", "Nesta Vector DB"),
            "element_type": "radio",
        },
    }
)

WIDGET_SPEC["knowledge_source"]["use_retrieval_option"] = WIDGET_SPEC["knowledge_source"]["options"][1]


def chat_history() -> List[BaseMessage]:
    """Derive chat history from streamlit messages"""

    if len(st.session_state.docgen["messages"]) > 1:  # omit initial_message from chat history
        return st.session_state.docgen["messages"][1:]

    return []


def send_revision_instructions(partial_state: State, *args) -> None:
    """Update the graph with the revision instructions and resume the graph"""

    partial_state["critique"] = st.session_state["revision_instructions"]
    partial_state = research_agent_revise(partial_state)
    preview_with_revision_option(*args, partial_state=partial_state)


def fill_preview_container(
    partial_state: State, pill_container: DeltaGenerator, preview_container: DeltaGenerator
) -> None:
    """Fill the preview container with the draft"""

    draft = markdown.markdown(partial_state["draft"])
    with preview_container.container():
        st.markdown(
            f"""
                **Preview**
                <div style="border:1px solid #ccc; padding:1rem; height:{PREVIEW_CONTAINER_HEIGHT}px; overflow:auto; background-color:#fafafa">
                    {draft}
                </div>
                """,  # noqa
            unsafe_allow_html=True,
        )

        st.text_area("Please provide any revision instructions, if needed", key="revision_instructions")

        st.button(
            "Submit",
            key="revision_submit",
            on_click=send_revision_instructions,
            args=(partial_state, *(pill_container, preview_container)),
        )


def preview_with_revision_option(pill_container: DeltaGenerator, *args, partial_state: Optional[State] = None) -> None:
    """Preview the draft and offer the option to add revision instructions"""

    if not partial_state:
        snapshot = graph.get_state(config)  # this only works because a second checkpoint has been set
        partial_state = snapshot.values

    fill_preview_container(partial_state, pill_container, *args)

    pill_container.pills(
        "Alternatively, upload to Google Drive?",
        ("Yes", "No"),
        key="upload",
        on_change=upload_and_complete_graph,
        args=(graph,),
    )


def upload_and_complete_graph(graph: CompiledStateGraph) -> None:
    """Upload the final draft to Google Drive (if the user confirms) and complete the graph"""

    graph.update_state(config, {"upload_confirmed": st.session_state["upload"].lower() == "yes"})
    graph.invoke(None, config)
    st.session_state.docgen["checkpoints_cleared"][1] = True
    st.session_state.docgen["messages"].pop(-1)  # remove the human message to allow a new request


def update_graph_and_resume(partial_state: State, pill_container: DeltaGenerator, *args) -> None:
    """Update the graph with the user's response to the template check and continue the graph to the next checkpoint"""

    if st.session_state["template_check"].lower() == "no":
        partial_state["intermediate_outputs"]["file_ids"] = None
        graph.update_state(config, partial_state)

    pill_container.empty()
    graph.invoke(None, config)
    st.session_state.docgen["checkpoints_cleared"][0] = True
    preview_with_revision_option(pill_container, *args)  # Action for second checkpoint


def check_template(pill_container: DeltaGenerator, *args) -> None:
    """Check whether the user should be applying the template"""

    def get_template_title(partial_state: State) -> str:
        file_ids_dict = partial_state["intermediate_outputs"].get("file_ids")
        _, file_ids = list(file_ids_dict.items())[
            -1
        ]  # see BaseDriveDoc.decide_whether_needs_document for a description of this dict's structure
        template_file_id = file_ids[0]  # there should be only one file_id in the list
        return OfficeTemplate.list_as_dict()[template_file_id].title

    snapshot = graph.get_state(config)  # this only works because a checkpoint has been set
    partial_state = snapshot.values

    check_template_message_format = (
        'Based on your request, I think I should be applying the following template: "{template_title}". '
        "Is that correct?"
    )
    check_template_message = check_template_message_format.format(template_title=get_template_title(partial_state))

    pill_container.pills(
        check_template_message,
        ("Yes", "No"),
        key="template_check",
        on_change=update_graph_and_resume,
        args=(partial_state, pill_container) + args,
    )


if __name__ == "__main__":

    # settings
    limit: int = 10
    stream: bool = False
    add_checkpoints: bool = True
    initial_message: str = "Hi, how can I help?"

    load_dotenv()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    graph, interrupt_before = create_graph(add_checkpoints=add_checkpoints)

    if st.session_state["connected"]:

        #        st.set_page_config(layout="wide")

        st.markdown(
            """
        <style>
            p {
                margin-bottom: 0;
            }

            a{
                margin-top: 0;
            }

            button {
                vertical-align: middle;
            }

            .response {
                margin: 25px 0 0 0;
                background-color: light-grey;
            }

        </style>
        """,
            unsafe_allow_html=True,
        )

        if DEBUG_MODE:
            st.markdown(
                '<p style="color:red;font-size:125%"><b>WARNING: DEBUG MODE IS ON</b></p>', unsafe_allow_html=True
            )

        st.markdown(
            "INTRO",
            unsafe_allow_html=True,
        )

        # widgets for filter conditions
        with st.sidebar:

            sidebar(WIDGET_SPEC)

            for key, spec in WIDGET_SPEC.items():
                if key not in st.session_state:
                    st.session_state[key] = spec["default"]

        # Store session variables
        if "docgen" not in st.session_state.keys():
            st.session_state["docgen"] = {}
            st.session_state["docgen"]["messages"] = [
                BaseMessage(content=initial_message, type="", role="assistant"),
            ]

        if add_checkpoints and not st.session_state.docgen.get("checkpoints_cleared"):
            st.session_state.docgen["checkpoints_cleared"] = [False, False]

        # Display chat messages
        for message in st.session_state.docgen["messages"]:
            with st.chat_message(message.role):
                if isinstance(message, CustomAIMessage):
                    st.markdown(message.as_html(), unsafe_allow_html=True)
                else:
                    st.write(message.content)

        # User-provided input
        if input := st.chat_input():
            st.session_state.docgen["messages"].append(HumanMessage(content=input, role="user"))
            with st.chat_message("user"):
                st.write(input)

        if isinstance(st.session_state.docgen["messages"][-1], HumanMessage) and (
            not add_checkpoints or not all(st.session_state.docgen["checkpoints_cleared"])
        ):

            with st.chat_message("assistant"):

                if not add_checkpoints or sum(st.session_state.docgen["checkpoints_cleared"]) == 0:

                    input = {
                        "messages": chat_history(),  # probably don't need this?
                        "limit": limit,
                        "use_hybrid_search": True,
                        "sidebar_options": {key: st.session_state[key] for key in WIDGET_SPEC.keys()},
                    }

                    graph_state = graph.invoke(
                        input, config=config, interrupt_before=interrupt_before
                    )  # NB config has no langfuse instructions at the moment

                    if add_checkpoints and sum(st.session_state.docgen["checkpoints_cleared"]) < 2:
                        preview_container = st.empty()
                        pill_container = st.empty()
                        check_template(*(pill_container, preview_container))  # Action for first checkpoint
