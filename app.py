import asyncio
import logging
import os
import re
import sys
import uuid

from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document as LangchainDocument
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables.base import Runnable
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from llm.tool import rag_chain_with_citation_tool


if (
    "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages" in sys.path
):  # streamlit seems to not like poetry; I had to add these three lines to get it to work
    sys.path.remove("/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages")
sys.path.append(
    "/Users/helen/Library/Caches/pypoetry/virtualenvs/dsp-nesta-brain-2RPY-0NE-py3.11/lib/python3.11/site-packages/"
)
import lxml.html  # noqa # nosec
import streamlit as st  # noqa

from dotenv import load_dotenv  # noqa
from dsp_nesta_brain import logger  # noqa
from langchain.chains import LLMChain  # noqa
from langchain_openai import ChatOpenAI  # noqa
from llm.prompt import basic_question_prompt  # noqa
from llm.prompt import contextualize_q_prompt  # noqa
from llm.prompt import qa_prompt  # noqa
from retrieval.retrieve import CustomRetriever  # noqa
from streamlit.delta_generator import DeltaGenerator  # noqa
from streamlit_feedback import streamlit_feedback  # noqa


langfuse = Langfuse()

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    user_id="anon",
)


EARLIEST_YEAR = 2003  # 2003 is the earliest publication date in the DB
DEFAULT_START_YEAR = 2019
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
        test_mode = False
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
    """A class just to make things like printing and writing to streamlit easier"""

    text: str
    quoted_answer: Optional[Dict] = None  # store quoted_answer instance if a tool has been used to derived citations
    mode: str
    references: List[Reference]
    index: Optional[int] = None
    trace_id: Optional[str] = None  # may need trace ids to push feedback to Langfuse

    def __init__(
        self,
        chain_response: Union[str, Dict],
        chunks: Union[LangchainDocument, List[LangchainDocument]],
        mode: str,
        index: Optional[int] = None,
    ) -> None:

        if mode == "chat":
            if type(chain_response["answer"]) is str:
                self.text = chain_response["answer"]
            elif isinstance(
                chain_response["answer"], dict
            ):  # this will be the case if a tool has been used for citations:
                # see quoted_answer class in llm/tool.py
                # use isinstance, not type
                self.quoted_answer = chain_response["answer"]["quoted_answer"]
                self.text = self.quoted_answer["answer"]
            chunks = chain_response["context"]
        elif mode == "indiv":
            self.text = chain_response.replace("ANSWER: ", "")
            if isinstance(chunks, LangchainDocument):
                chunks = [chunks]
        self.references = [Reference(chunk, i + 1) for i, chunk in enumerate(chunks)]
        self.index = index
        self.mode = mode

    def __repr__(self) -> str:
        """Self-explanatory"""
        string = "\n--------------\n" + self.text
        if not self.is_summary:
            string += f'\n{self.references[0].chunk.page_content}\n{self.references[0].metadata["location"]}'
        string += "\n--------------\n\n"
        return string

    @property
    def a_elements(self) -> str:
        """Return hyperlink(s) to source document(s)"""
        return [reference.as_html() for reference in self.references]

    @property
    def citations_in_text(self) -> List[str]:
        """Return the list of citations in the text, i.e. numbers appearing in square brackets"""
        return set(re.findall(r"\[\d+\]", self.text))

    @property
    def cited_references(self) -> List[Reference]:
        """Return a list of references which are actually cited in the text"""
        return [reference for reference in self.references if reference.cited]

    @property
    def p_element(self) -> str:
        """Return response text as an HTML paragraph"""
        header = "<b>SUMMARY:</b> " if self.is_summary else (f"({self.index}) " if self.index else "")
        return f"<p>{header}{self.text_with_superscript_citations}</p>"

    @property
    def is_summary(self) -> bool:
        """Determine whether the response should be treated as a summary of other visible responses"""
        return self.mode == "indiv" and self.index is None

    @property
    def references_(self) -> str:
        """Return formatted reference list"""
        if split_references:
            cited = [reference.as_html(reset_index=True) for reference in self.cited_references]
            not_cited = [reference.as_html(reset_index=True) for reference in self.uncited_references]
            actual_references = "<br><br><em>Cited references:</em><br>" + "<br>".join(cited) if cited else ""
            the_rest = (
                # Quick fix to show "May be useful" for uncited references in all situations
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
        css_class = "response " + ("summary" if self.is_summary else "indiv")
        return f'<div class="{css_class}">{self.p_element}{self.references_}</div>'

    def reset_reference_indices(self) -> None:
        """Reset how the reference numbering will appear if references are split into cited and uncited sources"""
        for citation in self.citations_in_text:
            citation_index = int(citation[1:-1])
            reference = self.references[citation_index - 1]
            reference.cited = True

        for i, reference in enumerate(self.cited_references + self.uncited_references):
            reference.reset_index = i + 1


def chat_history(*args) -> List[BaseMessage]:
    """
    Derive chat history from streamlit messages

    args are unused, but necessary if using chat_history as an argument in rag_chain_with_citation_tool to avoid an error
    """

    def message_class(message: Dict) -> type:
        return AIMessage if message["role"] == "assistant" else HumanMessage

    if (
        "messages" in st.session_state
    ):  # also necessary if using chat_history as an argument in rag_chain_with_citation_tool to avoid an error
        return [message_class(message)(content=msg["content"]) for msg in st.session_state.messages[1:]]
    else:
        return []


def trace_metadata() -> Dict:
    """Compile trace metadata on sidebar parameters and the resulting filter_condition string"""
    sidebar_metadata = {key: st.session_state[key] for key in WIDGET_DEFAULTS.keys()}
    metadata = {"sidebar": sidebar_metadata}
    metadata["retriever_filter_condition"] = st.session_state["filter_condition"]
    metadata["settings"] = {
        "merge": merge,
        "use_tool_for_citations": use_tool_for_citations,
        "limit": limit,
        "mode": mode,
    }
    return metadata


def llm_response(
    chain: LLMChain,
    docs: List[LangchainDocument],
    question: str,
    mode: str,
    message_placeholder: DeltaGenerator,
    **kwargs,
) -> str:
    """Get synchronous LLM response from chain"""
    if mode == "chat":
        input = {"input": question, "chat_history": chat_history()}
    else:
        input = {"context": docs, "question": question}
    trace_id = str(uuid.uuid4())
    response = {"answer": ""}

    for chunk in chain.stream(input, config={"run_id": trace_id, "callbacks": [langfuse_handler]}):
        # Process each chunk
        if "answer" in chunk:
            response_text = chunk["answer"]
            response["answer"] += str(response_text)
            # Display the response
            message_placeholder.markdown(response["answer"] + "▌")
        elif "context" in chunk:
            response["context"] = chunk["context"]
    # Remove the message placeholder text after all the text has been received, as
    # it will be rendered in a nicer format with references
    message_placeholder.markdown("")
    langfuse.trace(id=trace_id, metadata=trace_metadata())
    return response, trace_id


async def async_llm_response(chain: LLMChain, docs: List[LangchainDocument], question: str, **kwargs) -> str:
    """Get asynchronous LLM response from chain"""
    # NB: Async streaming is not implemented for now
    input = {"context": docs, "question": question}
    trace_id = str(uuid.uuid4())
    response = await chain.ainvoke(input, config={"run_id": trace_id, "callbacks": [langfuse_handler]}, **kwargs)
    langfuse.trace(id=trace_id, metadata=trace_metadata())
    return response, trace_id


async def individual_responses(chain: LLMChain, docs: List[LangchainDocument], question: str, **kwargs) -> List[str]:
    """Get asynchronous LLM responses for a number of documents/chunks from chain"""
    tasks = [asyncio.create_task(async_llm_response(chain, [doc], question, **kwargs)) for doc in docs]
    responses_and_trace_ids = await asyncio.gather(*tasks)
    return responses_and_trace_ids


def respond(
    chain: Runnable,
    docs: List[LangchainDocument],
    question: str,
    mode: str,
    message_placeholder: DeltaGenerator,
    **kwargs,
) -> List[Response]:
    """Get individual and/or summary responses from chain and convert them into Response objects"""

    responses = []

    if mode == "indiv":

        responses_and_trace_ids = asyncio.run(individual_responses(chain, docs, question, **kwargs))
        responses_ = [
            (response, docs[i]) for i, (response, _) in enumerate(responses_and_trace_ids) if response != "NULL"
        ]
        responses += [Response(response, doc, mode, index=i + 1) for i, (response, doc) in enumerate(responses_)]

    chain_response, trace_id = llm_response(chain, docs, question, mode, message_placeholder)

    if chain_response["answer"] != "NULL":
        responses.append(Response(chain_response, docs, mode))
    st.session_state["current_trace_id"] = trace_id

    # for response in enumerate(responses):
    #    logger.info(response)

    return responses


def is_html(string: str) -> bool:
    """Test whether a string is HTML"""
    # credit: https://stackoverflow.com/questions/24856035/how-to-detect-with-python-if-the-string-contains-html-code
    return lxml.html.fromstring(string).find(".//*") is not None


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

    if check_password():
        load_dotenv()
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # settings
        # retrieval settings
        possible_modes = ["chat", "indiv"]
        if sys.argv[1:] and sys.argv[1] in ["chat", "indiv"]:
            mode = sys.argv[1]
        else:
            mode = "chat"  # mode is either 'chat' for a chat with memeory or 'indiv' to return one response per doc
        merge = True  # merge needs to be True from now own for indexed references and inline citations to work
        # - otherwise we could get the same source reference appearing more than once in the reference list
        limit = 10
        use_tool_for_citations = False
        split_references = True  # if True, references will be split into cited and uncited retrieved sources
        # and the numbering reset so that references are numbered in the order they appear in the final list

        if mode not in possible_modes:
            raise Exception('Mode must be "chat" or "indiv"')

        llm = ChatOpenAI(
            temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini", streaming=True
        )
        indiv_qa_chain = create_stuff_documents_chain(llm, basic_question_prompt)
        chat_qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        retriever = CustomRetriever(
            merge=merge
        )  # merge cannot be passed through to the retriever via rag_chain kwargs, so set here
        # credit: https://medium.com/@eric_vaillancourt/mastering-langchain-rag-integrating-chat-history-part-2-4c80eae11b43
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        if use_tool_for_citations:
            rag_chain = rag_chain_with_citation_tool(history_aware_retriever, llm, qa_prompt, chat_history)
        else:
            rag_chain = create_retrieval_chain(history_aware_retriever, chat_qa_chain)

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
            }

            .indiv {
                background-color: light-grey;
            }

            .summary {
                border-style: solid;
                border-width: 1px;
                border-radius: 5px;
                background-color: #fae5af;
                border-color: #fae5af;
            }

        </style>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            # f"<h2>Demo (mode = '{mode}')</h2>",
            """
            <h2>🧠 Nesta Brain</h2><br/>This is an experimental prototype of a chatbot that "knows" a lot about Nesta.
            When you ask a question, it searches through thousands of webpages and reports, to find the most relevant content.
            <br/><br/>
            We hope this could be helpful for our knowledge management, such as for quickly finding information about
            our past projects and synthesising it into new outputs.
            The chatbot has access to information and reports on Nesta's website up to October 2024.
             </br></br>
            Use the parameters in the side bar to customise the information accessible to the chatbot (eg, select
            specific data range or mission team).</br></br>
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
                {"role": "assistant", "content": "Hi, how can I help?"},
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

                filter_condition = filter_conditions()
                retriever.filter_condition = filter_condition  # this is not ideal syntax, but kwargs to chain.invoke
                # are not passed on to the retriever
                st.session_state["filter_condition"] = filter_condition

                if mode == "indiv":
                    if input:
                        with st.spinner("Fetching documents ..."):
                            chunks = retriever.invoke(input, limit=limit, enumerate=True)
                else:
                    chunks = []  # if mode == 'chat', retrieval is already part of the chain

                if mode == "chat" or (mode == "indiv" and chunks):

                    responses = respond(
                        rag_chain if mode == "chat" else indiv_qa_chain, chunks, input, mode, message_placeholder
                    )

                if responses:
                    for response in responses:
                        message_placeholder.markdown(response.as_html(), unsafe_allow_html=True)
                        message = {"role": "assistant", "html": response.as_html(), "content": response.text}
                        st.session_state.messages.append(message)

                else:
                    st.write("I was not able to answer that question")

        # if there is more than one response, the feedback will be pushed to Langfuse with the trace_id of the last one
        feedback = streamlit_feedback(
            feedback_type="faces",
            optional_text_label="[Optional] Please provide an explanation",
            key="feedback",
            on_submit=push_feedback_to_langfuse,
        )
