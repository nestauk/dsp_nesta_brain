from __future__ import annotations

import logging
import os
import re
import uuid

from datetime import datetime
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import streamlit as st

from config import DEFAULT_START_YEAR
from config import EARLIEST_YEAR
from dotenv import load_dotenv
from dsp_nesta_brain import logger
from langchain.docstore.document import Document as LangchainDocument
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables.base import Runnable
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

# from llm.chain import history_aware_rag_chain
# from llm.chain import history_aware_rag_chain_with_citation_tool
from lgraph.graph import create_chat_graph
from streamlit.delta_generator import DeltaGenerator
from streamlit_feedback import streamlit_feedback


if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


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


class Reference:
    """A class to make inline citations easier"""

    chunk: LangchainDocument
    index: int
    reset_index: Optional[int] = None
    cited: bool = False

    def __init__(self, chunk: LangchainDocument, index: int) -> None:
        self.chunk = chunk
        self.index = index

    @property
    def is_pdf(self) -> bool:
        """Test whether the underlying source document is a PDF"""
        return self.metadata["location"].lower()[-4:] == ".pdf"

    @property
    def metadata(self) -> Dict:
        """Get chunk metadata"""
        return self.chunk.metadata

    def as_html(self, reset_index: bool = False) -> str:
        """Return reference metadata as an anchor element (indexed)"""
        test_mode = True
        index = self.reset_index if reset_index else self.index
        if test_mode:
            if self.index == 1:
                logger.warning(
                    "Formatting of links for testing retrieval filtering is in use – do not use for production"
                )
            return f'<a href="{self.metadata["location"]}">[{index}] {self.metadata["title"]}{" (PDF)" if self.is_pdf else ""} {self.metadata["date_pub"]} {self.metadata["contentType"]} {self.metadata["missions"]}</a>'  # noqa
        else:
            return f'<a href="{self.metadata["location"]}">[{index}] {self.metadata["title"]}{" (PDF)" if self.is_pdf else ""}</a>'  # noqa

    def as_superscript(self, reset_index: bool = False) -> str:
        """Return index as a clickable link within a superscript, suitable for inline citations"""
        return (
            f'<sup><a href="{self.metadata["location"]}">{self.reset_index if reset_index else self.index}</a></sup>'
        )


class Response:
    """A class just to make things like printing and writing responses to streamlit easier"""

    text: str
    references: List[Reference]
    trace_id: Optional[str] = None  # may need trace ids to push feedback to Langfuse

    def __init__(self, chain_response: Dict) -> None:

        if type(chain_response["response"]["answer"]) is str:
            self.text = chain_response["response"]["answer"]
        elif isinstance(
            chain_response["response"]["answer"], dict
        ):  # this will be the case if a tool has been used for citations:
            # see quoted_answer class in llm/tool.py
            # use isinstance, not type
            self.text = chain_response["response"]["answer"]["quoted_answer"]
        chunks = chain_response["response"]["context"]

        self.references = [Reference(chunk, i + 1) for i, chunk in enumerate(chunks)]

    def __repr__(self) -> str:
        """Self-explanatory"""
        string = "\n--------------\n" + self.text
        string += f'\n{self.references[0].chunk.page_content}\n{self.references[0].metadata["location"]}'
        string += "\n--------------\n\n"
        return string

    @property
    def a_elements(self) -> str:
        """Return hyperlink(s) to source document(s)"""
        return [reference.as_html() for reference in self.references]

    @property
    def citations_in_text(self) -> List[str]:
        """Return the set of citations in the text, i.e. numbers appearing in square brackets"""
        return set(re.findall(r"\[\d+\]", self.text))

    @property
    def cited_references(self) -> List[Reference]:
        """Return a list of references which are actually cited in the text"""
        return [reference for reference in self.references if reference.cited]

    @property
    def p_element(self) -> str:
        """Return response text as an HTML paragraph"""
        return f"<p>{self.text_with_superscript_citations}</p>"

    @property
    def references_(self) -> str:
        """Return formatted reference list"""
        if split_references:
            cited = [reference.as_html(reset_index=True) for reference in self.cited_references]
            not_cited = [reference.as_html(reset_index=True) for reference in self.uncited_references]
            actual_references = "<br><br><em>Cited references:</em><br>" + "<br>".join(cited) if cited else ""
            the_rest = (
                f"<br><em>{'May be useful' if cited else 'May be useful'}:</em><br>" + "<br>".join(not_cited)
                if not_cited
                else ""
            )
            return actual_references + the_rest
        else:
            a_elements = [reference.as_html() for reference in self.references]
            return "<br><br><em>Sources:</em><br>" + "<br>".join(a_elements)

    @property
    def text_with_superscript_citations(self) -> str:
        """
        Return text converting all citations in square brackets to a clickable superscript

        Note: we may encounter problems if for some reason numbers within square brackets appear in the text
        because they are part of the answer
        """

        if split_references:
            self.reset_reference_indices()

        text = self.text
        N_references = len(self.references)
        for citation in self.citations_in_text:
            citation_index = int(citation[1:-1])  # remove the square brackets
            if citation_index <= N_references:  # citation indices are in the range 1:N rather than 0:(N-1)
                reference = self.references[citation_index - 1]
                superscript = reference.as_superscript(reset_index=split_references)
                text = text.replace(citation, superscript)
            else:
                logging.warning(f"Citation {citation} contained an index greater than the number of references")
        text = text.replace(
            "</sup><sup>", ","
        )  # where there are citations next to each other, merge them into the same superscript and separate them with commas
        return text

    @property
    def uncited_references(self) -> List[Reference]:
        """Return a list of references which are not cited in the text"""
        return [reference for reference in self.references if not reference.cited]

    def as_html(self) -> str:
        """Convert the response into HTML"""
        return f'<div class="response">{self.p_element}{self.references_}</div>'

    def reset_reference_indices(self) -> None:
        """Reset how the reference numbering will appear if references are split into cited and uncited sources"""
        for citation in self.citations_in_text:
            citation_index = int(citation[1:-1])
            reference = self.references[citation_index - 1]
            reference.cited = True

        for i, reference in enumerate(self.cited_references + self.uncited_references):
            reference.reset_index = i + 1


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
        "limit": limit,
    }
    return metadata


def respond(
    chain: Runnable,
    question: str,
    message_placeholder: DeltaGenerator,
    **kwargs,
) -> Response:
    """Get synchronous LLM response from chain and convert it into a Response object"""

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

    if True:
        response = chain.invoke(input, config=config)  # ,stream_mode="custom"):

    else:
        pass
    # async for event in chain.astream_events(input, config,version="v1",stream_mode="values"):
    # Get chat model tokens from a particular node
    #    if event["event"] in ["on_chat_model_stream","on_chain_end"]:
    #       print(event)

    if use_langfuse:
        langfuse.trace(id=trace_id, metadata=trace_metadata())
        st.session_state["current_trace_id"] = trace_id

    return Response(response)


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
    use_langfuse: bool = False
    # retrieval settings
    # use_langgraph: bool = False
    merge: bool = True  # merge needs to be True from now on for indexed references and inline citations to work
    # - otherwise we could get the same source reference appearing more than once in the reference list
    limit: int = 10
    use_tool_for_citations: bool = False
    split_references: bool = True  # if True, references will be split into cited and uncited retrieved sources
    # and the numbering reset so that references are numbered in the order they appear in the final list

    # UI settings
    initial_message: str = "Hi, how can I help?"

    if check_password():
        load_dotenv()
        logging.getLogger("httpx").setLevel(logging.WARNING)

        rag_chain = create_chat_graph()

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

                response = respond(rag_chain, input, message_placeholder)
                message_placeholder.markdown(response.as_html(), unsafe_allow_html=True)
                message = {"role": "assistant", "html": response.as_html(), "content": response.text}
                st.session_state.messages.append(message)

        if use_langfuse:
            feedback = streamlit_feedback(
                feedback_type="faces",
                optional_text_label="[Optional] Please provide an explanation",
                key="feedback",
                on_submit=push_feedback_to_langfuse,
            )
