from __future__ import annotations

import asyncio
import logging
import os
import uuid

from datetime import datetime
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import streamlit as st

from config import ALLOW_POLICY_DOCS
from config import DEBUG_MODE
from config import DEPLOY_MODE
from config import EARLIEST_YEAR
from config import PROJECT
from config import USE_LANGFUSE
from dotenv import load_dotenv
from dsp_nesta_brain import logger
from front_end.auth.authenticate import Authenticator
from front_end.project_spec import INTRO
from front_end.project_spec import WIDGET_SPEC
from front_end.sidebar import sidebar
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables.base import Runnable
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from lgraph.graph import graph_options_type
from llm.chain import get_graph_or_rag_chain
from llm.message import CustomAIMessage
from streamlit.delta_generator import DeltaGenerator
from streamlit_feedback import streamlit_feedback


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

        return "seq:step:2" in (self.get("tags") or [])

    @property
    def stream(self) -> bool:
        """Test whether the AIMessageChunks in this event should be streamed"""
        return (
            self["event"] == "on_chat_model_stream"
            and (self.get("metadata") or {}).get("langgraph_node") in stream_nodes
            and not self.is_interim_message
        )


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
        "use_tool_for_citations": use_tool_for_citations,
        "use_graph": use_graph,
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
    }

    if use_graph in ["chat", "combined"]:

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

    else:

        if stream:

            message_text = ""
            for item in chain_or_graph.stream(input, config=config):
                # Process each item
                if "answer" in item:
                    if use_tool_for_citations:
                        item_text = (
                            item["answer"]["quoted_answer"].get("answer") or ""
                        )  # if using tool the answer will be a dict rather than string
                        if item_text and item_text == message_text:
                            break  # Once the response has been generated it will go on to the other components
                            # of quoted_answer which we don't actually need, so stop when the answer is complete
                        message_text += item_text[
                            len(message_text) :
                        ]  # unlike normal streaming, message_text contains the *cumulative* response
                        # this simulates normal streaming
                        # we could set response["answer"] = message_text, but I found this made the streaming look jerky
                    else:
                        message_text += str(item["answer"])
                    # Display the response
                    message_placeholder.markdown(message_text + "▌")

                elif "context" in item:
                    context = item["context"]

            response = {"answer": message_text, "context": context}

        else:
            response = chain_or_graph.invoke(input, config=config)

        return_message = CustomAIMessage(response)

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
    if ALLOW_POLICY_DOCS:
        use_graph: Optional[graph_options_type] = "combined"  # or None for none of the options
    else:
        use_graph = None
    use_langfuse: bool = (
        not DEBUG_MODE and PROJECT == "NESTA_BRAIN"
    )  # Langfuse is not currently set up for other projects –
    # don't want NestaBrain's Langfuse to store traces from other projects

    stream: bool = True
    use_tool_for_citations: bool = False

    # UI settings
    initial_message: str = "Hi, how can I help?"

    if use_tool_for_citations:
        raise Exception("use_tool_for_citations may no longer work – need to check")

    runnable, stream_nodes = get_graph_or_rag_chain(
        use_graph=use_graph, use_tool_for_citations=use_tool_for_citations, return_stream_nodes=True
    )

    load_dotenv()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    st.set_page_config(layout="wide")

    # -------authentication credit------
    # credit: https://medium.com/@coding-otter
    # https://medium.com/@coding-otter/google-oauth-in-streamlit-a-solution-that-finally-works-for-me-a212a79fec30
    if DEPLOY_MODE:
        redirect_uri = "https://nesta-brain.dap-tools.uk/"
    else:
        redirect_uri = "http://localhost:8501/"
    authenticator = Authenticator(
        # allowed_users=allowed_users,   #adapted to allow any email address with a nesta.org.uk domain
        token_key=os.getenv("AUTH_TOKEN_KEY"),
        secret_path="client_secret.json",  # nosec
        redirect_uri=redirect_uri,
    )

    authenticator.check_auth()
    authenticator.login()

    if st.session_state["connected"]:

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
            INTRO,
            unsafe_allow_html=True,
        )

        # widgets for filter conditions
        with st.sidebar:

            sidebar()

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

                response = respond(runnable, message_placeholder)
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
