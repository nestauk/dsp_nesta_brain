from __future__ import annotations

import asyncio
import os
import uuid

from datetime import datetime
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import streamlit as st

from config import ALLOW_POLICY_DOCS
from config import DEBUG_MODE
from config import EARLIEST_YEAR
from config import USE_LANGFUSE
from dsp_nesta_brain import logger
from front_end.project_spec import PAGE_INTRO
from front_end.project_spec import WIDGET_SPEC
from front_end.sidebar import sidebar
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables.base import Runnable
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from lgraph.graph import graph_options_type
from llm.chain import get_graph_or_rag_chain
from llm.message import CustomAIMessage
from streamlit.delta_generator import DeltaGenerator
from Welcome import setup


# from streamlit_feedback import streamlit_feedback


if TYPE_CHECKING:
    from langchain_core.messages.ai import AIMessageChunk
    from retrieval.retrieve import RetrieverInput as State

CURRENT_YEAR = datetime.now().year


langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

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

        return "seq:step:2" in self.get("tags") or []

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

    if len(st.session_state.chatbot["messages"]) > 1:  # omit initial_message from chat history
        return st.session_state.chatbot["messages"][1:]

    return []


def langfuse_mode() -> Literal["consent", "no_consent", False]:
    """Test whether to use Langfuse for monitoring and evaluation"""
    if USE_LANGFUSE:
        consent_option = WIDGET_SPEC["monitoring"]["options"][WIDGET_SPEC["monitoring"]["consent_option_index"]]
        consent = st.session_state["monitoring"] == consent_option
        return "consent" if consent else "no_consent"
    return False


def get_langfuse_config() -> None:
    """
    Set up Langfuse for monitoring and evaluation, if appropriate.

    The config is only needed if the user consents and we are not using a graph with streaming
    """

    config = {}

    mode = langfuse_mode()
    if mode:

        trace_id = str(uuid.uuid4())
        st.session_state["current_trace_id"] = trace_id

        if mode == "consent":
            streaming_with_graph = stream and use_graph in ["chat", "combined"]
            if not streaming_with_graph:
                # if streaming with graph then add the trace manually at the end of the streaming process (see comment below)
                # note that this means a detailed breakdown of the trace by chain/graph component is not available in Langfuse
                # otherwise add the trace here and pass config through to the graph or chain
                config = {"run_id": trace_id, "callbacks": [langfuse_handler]}
                langfuse.trace(id=trace_id, metadata=trace_metadata())

        elif mode == "no_consent":
            # if the user does not consent to monitoring, still log that they didn't consent, but no other information
            langfuse.trace(id=trace_id, metadata={"consent": False})

    return config


def trace_metadata(**kwargs) -> Dict:
    """Compile trace metadata on sidebar parameters and the resulting filter_condition string, as well as settings"""
    sidebar_metadata = {
        key: st.session_state[key] for key in WIDGET_SPEC.keys() if key != "monitoring"
    }  # the metadata isn't needed if monitoring is not consented to
    metadata = {"sidebar": sidebar_metadata}
    # metadata["retriever_filter_condition"] = st.session_state["filter_condition"]
    # not needed in metadata if it is part of input
    metadata["settings"] = {
        "use_tool_for_citations": use_tool_for_citations,
        "use_graph": use_graph,
        #   "limit": limit,      #not needed in metadata if it is part of input
    }
    metadata[
        "policy_file_ids"
    ] = None  # this needs to be updated via kwargs after the graph has run, if the graph is used
    metadata.update(kwargs)
    return metadata


def respond(
    chain_or_graph: Runnable,
    message_placeholder: DeltaGenerator,
    **kwargs,
) -> CustomAIMessage:
    """Get LLM response from chain"""

    config = get_langfuse_config()

    input = {
        "messages": chat_history(),
        "filter_condition": st.session_state.chatbot["filter_condition"],
        "limit": limit,
        "use_hybrid_search": True,
    }

    if use_graph in ["chat", "combined"]:

        if stream:

            async def stream_() -> State:

                message_text = ""
                id = None

                async for event in chain_or_graph.astream_events(input, version="v1", stream_mode="values"):

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

            if langfuse_mode() == "consent":

                # the Langfuse trace is added manually here with the output because passing config
                # to .astream_events did not seem to work and resulted in blank outputs in traces
                policy_file_ids = final_state.get("intermediate_outputs", {}).get("policy_file_ids")
                output = {
                    k: v
                    for k, v in final_state["raw_response"].items()
                    if k
                    not in [
                        "limit",
                        "filter_condition",
                        "use_hybrid_search",
                        "intermediate_outputs",
                    ]  # either in input or not needed
                }
                langfuse.trace(
                    id=st.session_state["current_trace_id"],
                    input=input,
                    output=output,
                    metadata=trace_metadata(policy_file_ids=policy_file_ids),
                    user_id=os.getenv("LANGFUSE_USER_ID"),
                )

        else:
            final_state = chain_or_graph.invoke(input, config=config)
            if langfuse_mode() == "consent":
                policy_file_ids = final_state.get("intermediate_outputs", {}).get("policy_file_ids")
                langfuse.trace(
                    id=st.session_state["current_trace_id"], metadata={"policy_file_ids": policy_file_ids}
                )  # this will update just the relevant key-value pair in metadata

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

    return return_message


def filter_conditions() -> Union[str, None]:
    """Compute what the filter conditions are from widget values"""

    filter_conditions = []

    for key, spec in WIDGET_SPEC.items():

        if spec.get("filter_condition_format"):
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


def push_feedback_to_langfuse() -> None:
    """Send the feedback score and comments to Langfuse"""

    trace_id = st.session_state["current_trace_id"]

    #    faces_score_map = {"😞": 1, "🙁": 2, "😐": 3, "🙂": 4, "😀": 5}

    langfuse.score(
        # trace_id=trace_id, name="user-feedback", value=faces_score_map[feedback["score"]], comment=feedback["text"]
        trace_id=trace_id,
        name="user-feedback",
        value=st.session_state["feedback"],
        comment="N/A",
    )

    logger.info(f"Pushed user feedback for trace_id {trace_id} to Langfuse")


if __name__ == "__main__":

    # settings

    if ALLOW_POLICY_DOCS:
        use_graph: Optional[graph_options_type] = "combined"  # or None for none of the options
    else:
        use_graph = None

    limit: int = 10
    stream: bool = True
    use_tool_for_citations: bool = (
        False  # this is obsolete but retained in case future developers want to experiment with improving citations
    )
    if use_tool_for_citations:
        raise Exception("use_tool_for_citations is deprecated. Set to False")

    # UI settings
    initial_message: str = "Hi, how can I help?"

    setup()

    runnable, stream_nodes = get_graph_or_rag_chain(
        use_graph=use_graph, use_tool_for_citations=use_tool_for_citations, return_stream_nodes=True
    )

    if st.session_state["connected"]:

        runnable, stream_nodes = get_graph_or_rag_chain(
            use_graph=use_graph, use_tool_for_citations=use_tool_for_citations, return_stream_nodes=True
        )

        #   st.set_page_config(layout="wide")
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
            PAGE_INTRO,
            unsafe_allow_html=True,
        )

        # widgets for filter conditions
        with st.sidebar:

            sidebar()

            for key, spec in WIDGET_SPEC.items():
                if key not in st.session_state:
                    st.session_state[key] = spec["default"]

        # Store session variables
        if "chatbot" not in st.session_state.keys():
            st.session_state["chatbot"] = {}
            st.session_state["chatbot"]["messages"] = [
                BaseMessage(content=initial_message, type="", role="assistant"),
            ]

        # Display chat messages
        for message in st.session_state.chatbot["messages"]:
            with st.chat_message(message.role):
                if isinstance(message, CustomAIMessage):
                    st.markdown(message.as_html(), unsafe_allow_html=True)
                else:
                    st.write(message.content)

        # User-provided input
        if input := st.chat_input():
            st.session_state.chatbot["messages"].append(HumanMessage(content=input, role="user"))
            with st.chat_message("user"):
                st.write(input)

        # Generate a new response if last message is not from assistant
        if isinstance(st.session_state.chatbot["messages"][-1], HumanMessage):

            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                st.session_state.chatbot["filter_condition"] = filter_conditions()

                message = respond(runnable, message_placeholder)
                message_placeholder.markdown(message.as_html(), unsafe_allow_html=True)
                st.session_state.chatbot["messages"].append(message)

        if langfuse_mode() == "consent":
            # feedback = streamlit_feedback(
            #     feedback_type="faces",
            #     optional_text_label="[Optional] Please provide an explanation",
            #     key="feedback",
            #     on_submit=push_feedback_to_langfuse,
            # )
            feedback = st.feedback(
                options="faces",
                key="feedback",
                on_change=push_feedback_to_langfuse,
            )
            st.markdown(
                """
                <style>
                    div[aria-label="button group"] {
                        display: flex;
                        justify-content: flex-end;
                        max-width: 100% !important;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
