from __future__ import annotations

import asyncio
import logging
import os
import uuid

from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Union

import streamlit as st

from config import DEBUG_MODE
from config import EARLIEST_YEAR
from config import USE_LANGFUSE
from dotenv import load_dotenv
from dsp_nesta_brain import logger
from front_end.sidebar import sidebar
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables.base import Runnable
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from lgraph.research_agent.research_agent import agent
from llm.message import CustomAIMessage
from streamlit.delta_generator import DeltaGenerator
from streamlit_feedback import streamlit_feedback


stream_nodes = []  # temporary, to please Flake8

if TYPE_CHECKING:
    from langchain_core.messages.ai import AIMessageChunk
    from retrieval.retrieve import RetrieverInput as State

CURRENT_YEAR = datetime.now().year


langfuse = Langfuse()

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    user_id=os.getenv("LANGFUSE_USER_ID"),
)


WIDGET_SPEC = OrderedDict(
    {
        "knowledge_source": {
            "default": "Nesta Vector DB",
            "filter_condition_format": None,
            "options": ("LLM internal knowledge", "Nesta Vector DB"),
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

    def message_class(message: Dict) -> type:
        return AIMessage if message["role"] == "assistant" else HumanMessage

    if len(st.session_state.messages) > 1:  # omit initial_message from chat history
        return [message_class(msg)(content=msg["content"]) for msg in st.session_state.messages[1:]]

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


def respond(
    chain_or_graph: Runnable,
    message_placeholder: DeltaGenerator,
    **kwargs,
) -> CustomAIMessage:
    """Get LLM response from chain"""

    if USE_LANGFUSE:
        trace_id = str(uuid.uuid4())
        config = {"run_id": trace_id, "callbacks": [langfuse_handler]}
    else:
        config = {}

    input = {
        "messages": chat_history(),
        "filter_condition": st.session_state["filter_condition"],
        "limit": limit,
        "use_hybrid_search": True,
        "sidebar_options": {key: st.session_state[key] for key in WIDGET_SPEC.keys()},
    }

    if stream:

        async def stream_() -> State:

            message_text = ""
            id = None

            async for event in chain_or_graph.astream_events(input, config, version="v1", stream_mode="values"):

                event = GraphStreamEvent(event)

                if event.stream:
                    if id != event.ai_message_chunk.id:
                        if id:
                            message_text += "\n\n"
                    id = event.ai_message_chunk.id
                    message_text += event.ai_message_chunk.content
                    message_placeholder.markdown(message_text + "▌")

                elif event.return_final_state:
                    return event["data"]["input"]

        final_state = asyncio.run(stream_())

    else:
        final_state = chain_or_graph.invoke(input, config=config)

    return_message = final_state["messages"][-1]

    if stream:
        # Remove the message placeholder text after all the text has been received, as
        # it will be rendered in a nicer format with references
        message_placeholder.markdown("")

    if USE_LANGFUSE:
        langfuse.trace(id=trace_id, metadata=trace_metadata())
        st.session_state["current_trace_id"] = trace_id

    return return_message


def filter_conditions() -> Union[str, None]:
    """Compute what the filter conditions are from widget values"""

    filter_conditions = []

    for key, spec in WIDGET_SPEC.items():

        default = spec["default"]
        filter_condition_format = spec["filter_condition_format"]
        current_value = st.session_state[key]

        if key == "from_year":
            append_filter_condition = current_value != EARLIEST_YEAR
        else:
            append_filter_condition = current_value != default
            # caution: if the rest of the widgets are at their default value then no filter is required
            # if the defaults change, the logic here may also need to change

        if append_filter_condition:
            filter_conditions.append(filter_condition_format.format(current_value=current_value))

    if filter_conditions:
        return " and ".join(filter_conditions)

    return None


def push_feedback_to_langfuse(feedback: Dict) -> None:
    """Send the feedback score and comments to Langfuse"""

    trace_id = st.session_state["current_trace_id"]

    faces_score_map = {"😞": 1, "🙁": 2, "😐": 3, "🙂": 4, "😀": 5}

    langfuse.score(
        trace_id=trace_id, name="user-feedback", value=faces_score_map[feedback["score"]], comment=feedback["text"]
    )

    logger.info(f"Pushed user feedback for trace_id {trace_id} to Langfuse")


if __name__ == "__main__":

    # settings
    limit: int = 10
    stream: bool = False

    # UI settings
    initial_message: str = "Hi, how can I help?"

    if check_password():
        load_dotenv()
        logging.getLogger("httpx").setLevel(logging.WARNING)

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
                {"role": "assistant", "content": initial_message},
            ]

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message.get("html"):
                    st.markdown(message["html"], unsafe_allow_html=True)
                else:
                    st.write(message["content"])

        # User-provided input
        if input := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": input})
            with st.chat_message("user"):
                st.write(input)

        # Generate a new response if last message is not from assistant
        responses = []
        if st.session_state.messages[-1]["role"] != "assistant":

            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                st.session_state["filter_condition"] = filter_conditions()

                response = respond(agent, message_placeholder)
                message_placeholder.markdown(response.as_html(), unsafe_allow_html=True)
                message = {"role": "assistant", "html": response.as_html(), "content": response.content}
                st.session_state.messages.append(message)

        if USE_LANGFUSE:
            feedback = streamlit_feedback(
                feedback_type="faces",
                optional_text_label="[Optional] Please provide an explanation",
                key="feedback",
                on_submit=push_feedback_to_langfuse,
            )
