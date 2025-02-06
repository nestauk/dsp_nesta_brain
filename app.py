from __future__ import annotations

import asyncio
import logging
import os
import uuid

from datetime import datetime
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Union

import streamlit as st

from config import DEBUG_MODE
from config import DEFAULT_START_YEAR
from config import EARLIEST_YEAR
from dotenv import load_dotenv
from dsp_nesta_brain import logger
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables.base import Runnable
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from lgraph.graph import LAST_CHAT_GRAPH_NODE_NAME
from lgraph.graph import create_chat_graph
from llm.chain import history_aware_rag_chain

# from llm.chain import history_aware_rag_chain_with_citation_tool
from llm.message import CustomAIMessage
from streamlit.delta_generator import DeltaGenerator
from streamlit_feedback import streamlit_feedback


if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from retrieval.retrieve import RetrieverInput as State


langfuse = Langfuse()

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    user_id=os.getenv("LANGFUSE_USER_ID"),
)


CURRENT_YEAR = datetime.now().year

WIDGET_DEFAULTS = {"from_year": DEFAULT_START_YEAR, "to_year": CURRENT_YEAR, "include_people": "Yes", "mission": None}


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
    sidebar_metadata = {key: st.session_state[key] for key in WIDGET_DEFAULTS.keys()}
    metadata = {"sidebar": sidebar_metadata}
    metadata["retriever_filter_condition"] = st.session_state["filter_condition"]
    metadata["settings"] = {
        "merge": merge,
        "use_tool_for_citations": use_tool_for_citations,
        "use_graph": use_graph,
        "limit": limit,
    }
    return metadata


def respond(
    chain: Runnable,
    message_placeholder: DeltaGenerator,
    **kwargs,
) -> CustomAIMessage:
    """Get LLM response from chain"""

    if use_langfuse:
        trace_id = str(uuid.uuid4())
        config = {"run_id": trace_id, "callbacks": [langfuse_handler]}
    else:
        config = {}

    input = {
        "messages": chat_history(),
        "filter_condition": st.session_state["filter_condition"],
        "merge": merge,
    }

    if use_graph:

        if stream:

            async def stream_() -> State:
                message_text = ""
                id = None
                async for event in chain.astream_events(input, config, version="v1", stream_mode="values"):
                    if event["event"] == "on_chat_model_stream":
                        ai_message_chunk = event["data"]["chunk"]
                        if id != ai_message_chunk.id:
                            if id:
                                message_text += "\n\n"
                            id = ai_message_chunk.id
                        message_text += ai_message_chunk.content
                        message_placeholder.markdown(message_text + "▌")
                    elif event["event"] == "on_chain_end" and event["name"] == LAST_CHAT_GRAPH_NODE_NAME:
                        final_state = event["data"]["input"]
                return final_state

            final_state = asyncio.run(stream_())

        else:
            final_state = chain.invoke(input, config=config)

        return_message = final_state["messages"][-1]

    else:

        if stream:

            message_text = ""
            for item in chain.stream(input, config=config):
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
            response = chain.invoke(input, config=config)

        return_message = CustomAIMessage(response)

    if stream:
        # Remove the message placeholder text after all the text has been received, as
        # it will be rendered in a nicer format with references
        message_placeholder.markdown("")

    if use_langfuse:
        langfuse.trace(id=trace_id, metadata=trace_metadata())
        st.session_state["current_trace_id"] = trace_id

    return return_message


def filter_conditions() -> Union[str, None]:
    """Compute what the filter conditions are from widget values"""
    filter_conditions = []
    for key, default in WIDGET_DEFAULTS.items():
        current_value = st.session_state[key]
        if key == "from_year" and current_value != EARLIEST_YEAR:
            filter_conditions.append(f"source.date_pub >= to_timestamp('{current_value}-01-01')")
        elif (
            current_value != default
        ):  # caution: if the rest of the widgets are at their default value then no filter is required
            # if the defaults change, the logic here may also need to change
            if key == "to_year":
                filter_conditions.append(f"source.date_pub <= to_timestamp('{current_value}-12-31')")
            elif key == "include_people" and current_value == "No":
                filter_conditions.append("source.contentType != 'person page'")
            elif key == "mission":
                filter_conditions.append(f"array_contains(source.missions,'{current_value}')")
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
    use_graph: bool = True
    use_langfuse: bool = not DEBUG_MODE
    stream: bool = True
    # retrieval settings
    # use_langgraph: bool = False    #for simplification. This was previously the setting to use LangGraph for retrieval
    merge: bool = True  # merge needs to be True from now on for indexed references and inline citations to work
    # - otherwise we could get the same source reference appearing more than once in the reference list
    limit: int = 10
    use_tool_for_citations: bool = False

    # UI settings
    initial_message: str = "Hi, how can I help?"

    if check_password():
        load_dotenv()
        logging.getLogger("httpx").setLevel(logging.WARNING)

        if use_tool_for_citations:
            raise Exception("use_tool_for_citations may no longer work – need to check")

        if use_graph:
            rag_chain = create_chat_graph()
        else:
            rag_chain = history_aware_rag_chain()

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
            """
            <h2>🧠 Nesta Brain</h2><br/>
            This is a prototype AI chatbot designed to help you explore Nesta's knowledge.
            It searches thousands of webpages and reports to find the most relevant content
            in response to your questions.
            <br/><br/>
            We hope this can support knowledge management by making it easier to locate information
            about past projects,
            and generate new outputs.
            <br/><br/>
            This is an early version and we welcome your feedback
            very much - please use the
            emojis below to highlight specific responses, and <a href='https://forms.gle/TwXqUMHNTaPbYC4e7'>leave
            us general feedback using this form</a>.
            You can also contact directly Karlis Kanders or Helen Jackson (Data Science Practice / Discovery Hub)
            on <a href="https://nesta.slack.com/archives/C05BCUZNATG">#proj-nesta-brain</a>.
            <br/><br/>
            The chatbot currently accesses information from <strong>Nesta's public website (up to October 2024)</strong>
            and does <strong>not</strong> include internal documents or systems like Nesta:Net, Slack, or GitHub.
            <br/><br/>
            Use the sidebar to customize the chatbot's search parameters, such as date range or mission team.
            Note that user queries and responses are saved for chatbot's performance evaluation and improvement.
            """,
            unsafe_allow_html=True,
        )

        # widgets for filter conditions
        with st.sidebar:
            from_year = st.number_input(
                label="From year",
                min_value=EARLIEST_YEAR,
                max_value=CURRENT_YEAR,
                key="from_year",
                value=WIDGET_DEFAULTS["from_year"],
            )
            to_year = st.number_input(
                label="To year",
                min_value=from_year,
                max_value=CURRENT_YEAR,
                key="to_year",
                value=WIDGET_DEFAULTS["to_year"],
            )
            include_people_options = ("Yes", "No")
            include_people = st.radio(
                "Include people pages",
                include_people_options,
                key="include_people",
                index=include_people_options.index(WIDGET_DEFAULTS["include_people"]),
            )
            mission_options = ("A fairer start", "A healthy life", "A sustainable future", None)
            mission = st.radio(
                "Mission-specific content",
                mission_options,
                key="mission",
                index=mission_options.index(WIDGET_DEFAULTS["mission"]),
            )

            for key, default in WIDGET_DEFAULTS.items():
                if key not in st.session_state:
                    st.session_state[key] = default

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

                response = respond(rag_chain, message_placeholder)
                message_placeholder.markdown(response.as_html(), unsafe_allow_html=True)
                message = {"role": "assistant", "html": response.as_html(), "content": response.content}
                st.session_state.messages.append(message)

        if use_langfuse:
            feedback = streamlit_feedback(
                feedback_type="faces",
                optional_text_label="[Optional] Please provide an explanation",
                key="feedback",
                on_submit=push_feedback_to_langfuse,
            )
