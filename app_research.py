from __future__ import annotations

import logging
import os

from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Dict
from typing import List

import markdown
import streamlit as st

from config import DEBUG_MODE
from dotenv import load_dotenv
from dsp_nesta_brain import logger
from front_end.sidebar import sidebar
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from lgraph.research_agent.research_agent import create_graph
from llm.message import CustomAIMessage


stream_nodes = []  # temporary, to please Flake8

if TYPE_CHECKING:
    from langchain_core.messages.ai import AIMessageChunk

CURRENT_YEAR = datetime.now().year


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


class GraphStreamEvent(dict):
    """
    A class to represent the outputs of the astream_events method of graphs,
    in order to make the streaming syntax more readable
    """  # noqa

    @property
    def ai_message_chunk(self) -> AIMessageChunk:
        """Return the AI message chunk from the event"""
        return self["data"]["chunk"]

    @property
    def return_final_state(self) -> bool:
        """Test whether to return the final graph state"""
        return self["event"] == "on_chain_end" and self["name"] == stream_nodes[-1]

    @property
    def is_interim_message(self) -> bool:
        """
        Test whether the AIMessageChunks relate to the content of an InterimMessage (e.g. from recontextualisation),
        in which case it shouldn't be streamed

        The seq:step:N tag represents a step number in the execution sequence of different steps in the graph

        CAUTION!!!: if the structure of the graph or chains changes, the step number may change and this test may need to be updated
        """  # noqa

        return "seq:step:2" in self.get("tags") or []

    @property
    def stream(self) -> bool:
        """Test whether the AIMessageChunks in this event should be streamed"""
        condition_met = self["event"] == "on_chat_model_stream"
        condition_met = condition_met and (self.get("metadata") or {}).get("langgraph_node") in stream_nodes
        condition_met = condition_met and not self.is_interim_message
        return condition_met


def check_password() -> bool:
    """Return `True` if the user had the correct password."""

    def password_entered() -> None:
        """Check whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.

        return True


def chat_history() -> List[BaseMessage]:
    """Derive chat history from streamlit messages"""

    if len(st.session_state.messages) > 1:  # omit initial_message from chat history
        return st.session_state.messages[1:]

    return []


def trace_metadata() -> Dict:
    """Compile trace metadata on sidebar parameters and the resulting filter_condition string, as well as settings"""
    sidebar_metadata = {key: st.session_state[key] for key in WIDGET_SPEC.keys()}
    metadata = {"sidebar": sidebar_metadata}
    metadata["retriever_filter_condition"] = st.session_state["filter_condition"]
    metadata["settings"] = {
        "research_agent": True,
        "limit": limit,
    }
    return metadata


def push_feedback_to_langfuse(feedback: Dict) -> None:
    """Send the feedback score and comments to Langfuse"""

    trace_id = st.session_state["current_trace_id"]

    faces_score_map = {"😞": 1, "🙁": 2, "😐": 3, "🙂": 4, "😀": 5}

    langfuse.score(
        trace_id=trace_id, name="user-feedback", value=faces_score_map[feedback["score"]], comment=feedback["text"]
    )

    logger.info(f"Pushed user feedback for trace_id {trace_id} to Langfuse")


def update_agent_complete_graph() -> None:
    """Update the agent with the edited draft and complete the graph – intended as a callback for the text_area widget"""

    st.session_state["edited_draft"] = st.session_state.text_area
    agent.update_state(config, {"draft": st.session_state.edited_draft, "finalized_state": True, "edited": True})
    st.session_state.final_graph_state = agent.invoke(None, config)  # finish the graph after the checkpoint


if __name__ == "__main__":

    # settings
    limit: int = 10
    stream: bool = False
    editable: bool = False

    # UI settings
    initial_message: str = "Hi, how can I help?"

    if check_password():

        load_dotenv()
        logging.getLogger("httpx").setLevel(logging.WARNING)

        agent = create_graph(editable=editable)

        st.set_page_config(layout="wide")

        st.markdown(
            """
        <style>
            p {
                margin-bottom: 0;
            }

            a{
                margin-top: 0;
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
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                BaseMessage(content=initial_message, type="", role="assistant"),
            ]
        if "edited_draft" not in st.session_state.keys():
            st.session_state.edited_draft = None

        if "filter_condition" not in st.session_state.keys():
            st.session_state.filter_condition = None  # not needed at the moment

        if "final_graph_state" not in st.session_state.keys():
            st.session_state.final_graph_state = None

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message.role):
                if isinstance(message, CustomAIMessage):
                    st.markdown(message.as_html(), unsafe_allow_html=True)
                else:
                    st.write(message.content)

        # User-provided input
        if input := st.chat_input():
            st.session_state.messages.append(HumanMessage(content=input, role="user"))
            with st.chat_message("user"):
                st.write(input)

        if isinstance(st.session_state.messages[-1], HumanMessage) and not st.session_state.edited_draft:

            with st.chat_message("assistant"):

                input = {
                    "messages": chat_history(),
                    "filter_condition": st.session_state["filter_condition"],
                    "limit": limit,
                    "use_hybrid_search": True,
                    "sidebar_options": {key: st.session_state[key] for key in WIDGET_SPEC.keys()},
                }

                graph_state = agent.invoke(
                    input, config=config, interrupt_before="terminate" if editable else None
                )  # NB config has no langfuse instructions

                if editable:
                    snapshot = agent.get_state(config)  # this only works because a checkpoint has been set
                    partial_state = snapshot.values

                    with st.container():

                        col1, col2 = st.columns(2, gap="medium")
                        height = 500
                        draft = markdown.markdown(partial_state["draft"])

                        with col1:
                            st.markdown("\n**Preview**")
                            st.markdown(
                                f"""
                                    <div style="border:1px solid #ccc; padding:1rem; height:{height}px; overflow:auto; background-color:#fafafa">
                                        {draft}
                                    </div>
                                    """,  # noqa
                                unsafe_allow_html=True,
                            )

                        with col2:
                            st.markdown("\n**✍️ Edit Markdown**")
                            st.text_area(
                                "Report content",
                                value=partial_state["draft"],
                                height=height,
                                label_visibility="collapsed",
                                key="text_area",
                                on_change=update_agent_complete_graph,
                            )

                else:
                    st.session_state.final_graph_state = graph_state

        if st.session_state.final_graph_state:

            # final_snapshot = agent.get_state(config)  #graph should have been completed in update_agent_complete_graph
            # final_state = final_snapshot.values
            # print(final_snapshot)
            #  message = final_state['messages'][-1]

            message = st.session_state.final_graph_state["messages"][-1]
            st.markdown(message.as_html(), unsafe_allow_html=True)
            st.session_state.messages.append(message)
            #  st.session_state.edited_draft = None
            st.session_state.final_graph_state = None

        #   if USE_LANGFUSE:
        #      feedback = streamlit_feedback(
        #         feedback_type="faces",
        #        optional_text_label="[Optional] Please provide an explanation",
        #       key="feedback",
        #      on_submit=push_feedback_to_langfuse,
        # )
