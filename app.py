import asyncio
import logging
import os
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
from utils import unique


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
from streamlit_feedback import streamlit_feedback  # noqa


langfuse = Langfuse()

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    user_id="anon",
)


EARLIEST_YEAR = 2003  # 2003 is the earliest publication date in the DB
CURRENT_YEAR = datetime.now().year

WIDGET_DEFAULTS = {"from_year": EARLIEST_YEAR, "to_year": CURRENT_YEAR, "include_people": "Yes", "mission": None}


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


class Response:
    """A class just to make things like printing and writing to streamlit easier"""

    text: str
    mode: str
    chunks: List[LangchainDocument]
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
            text = chain_response["answer"]
            chunks = chain_response["context"]
        elif mode == "indiv":
            text = chain_response.replace("ANSWER: ", "")
            if isinstance(chunks, LangchainDocument):
                chunks = [chunks]
        self.text = text
        self.chunks = chunks
        self.index = index
        self.mode = mode

    def __repr__(self) -> str:
        """Self-explanatory"""
        string = "\n--------------\n" + self.text
        if not self.is_summary:
            string += f'\n{self.chunks[0].page_content}\n{self.chunks[0].metadata["location"]}'
        string += "\n--------------\n\n"
        return string

    @property
    def a_elements(self) -> str:
        """Return hyperlink(s) to source document(s)"""
        test_mode = False
        if test_mode:
            logger.warning("Formatting of links for testing retrieval filtering is in use – do not use for production")
            elements = [
                f'<a href="{chunk.metadata["location"]}">{chunk.metadata["title"]} {chunk.metadata["date_pub"]} {chunk.metadata["contentType"]} {chunk.metadata["missions"]}</a>'  # noqa
                for chunk in self.chunks
            ]
        else:
            elements = [
                f'<a href="{chunk.metadata["location"]}">{chunk.metadata["title"]}</a>' for chunk in self.chunks
            ]
        return "<br><br><em>References</em><br>" + "<br>".join(unique(elements))

    @property
    def p_element(self) -> str:
        """Return response text as an HTML paragraph"""
        return f'<p>{"<b>SUMMARY:</b> " if self.is_summary else (f"({self.index}) " if self.index else "")}{self.text}</p>'

    @property
    def is_summary(self) -> bool:
        """Determine whether the response should be treated as a summary of other visible responses"""
        return self.mode == "indiv" and self.index is None

    def as_html(self) -> str:
        """Convert the response into HTML"""
        css_class = "response " + ("summary" if self.is_summary else "indiv")
        return f'<div class="{css_class}">{self.p_element}{self.a_elements}</div>'


def chat_history() -> List[BaseMessage]:
    """Derive chat history from streamlit messages"""

    def message_class(message: Dict) -> type:
        return AIMessage if message["role"] == "assistant" else HumanMessage

    return [message_class(message)(content=msg["content"]) for msg in st.session_state.messages[1:]]


def llm_response(chain: LLMChain, docs: List[LangchainDocument], question: str, mode: str, **kwargs) -> str:
    """Get synchronous LLM response from chain"""
    if mode == "chat":
        input = {"input": question, "chat_history": chat_history()}
    else:
        input = {"context": docs, "question": question}
    trace_id = str(uuid.uuid4())
    response = chain.invoke(input, config={"run_id": trace_id, "callbacks": [langfuse_handler]}, *kwargs)
    return response, trace_id


async def async_llm_response(chain: LLMChain, docs: List[LangchainDocument], question: str, **kwargs) -> str:
    """Get asynchronous LLM response from chain"""
    #  print("message history",chat_history())
    #  input = {"input": question,"chat_history":chat_history()}
    input = {"context": docs, "question": question}
    trace_id = str(uuid.uuid4())
    response = await chain.ainvoke(input, config={"run_id": trace_id, "callbacks": [langfuse_handler]}, **kwargs)
    return response, trace_id


async def individual_responses(chain: LLMChain, docs: List[LangchainDocument], question: str, **kwargs) -> List[str]:
    """Get asynchronous LLM responses for a number of documents/chunks from chain"""
    tasks = [asyncio.create_task(async_llm_response(chain, [doc], question, **kwargs)) for doc in docs]
    responses_and_trace_ids = await asyncio.gather(*tasks)
    return responses_and_trace_ids


def respond(chain: Runnable, docs: List[LangchainDocument], question: str, mode: str, **kwargs) -> List[Response]:
    """Get individual and/or summary responses from chain and convert them into Response objects"""

    responses = []

    if mode == "indiv":

        responses_and_trace_ids = asyncio.run(individual_responses(chain, docs, question, **kwargs))
        responses_ = [
            (response, docs[i]) for i, (response, _) in enumerate(responses_and_trace_ids) if response != "NULL"
        ]
        responses += [Response(response, doc, mode, index=i + 1) for i, (response, doc) in enumerate(responses_)]

    response, trace_id = llm_response(chain, docs, question, mode)

    response = Response(response, docs, mode)
    if response.text != "NULL":
        responses.append(response)
    st.session_state["current_trace_id"] = trace_id

    # for response in enumerate(responses):
    #    logger.info(response)

    return responses


def is_html(string: str) -> bool:
    """Test whether a string is HTML"""
    # credit: https://stackoverflow.com/questions/24856035/how-to-detect-with-python-if-the-string-contains-html-code
    return lxml.html.fromstring(string).find(".//*") is not None


def filter_flag() -> None:
    """
    Indicate that filters have been used and therefore filter conditions need to be provided
    (used in widget callbacks)
    """  # noqa
    st.session_state.filter_flag = True


def filter_conditions() -> Union[str, None]:
    """Compute what the filter conditions are from widget values"""
    if "filter_flag" in st.session_state:
        filter_conditions = []
        for key, default in WIDGET_DEFAULTS.items():
            current_value = st.session_state[key]
            if (
                current_value != default
            ):  # caution: if all widgets are at their default value then no filter is required
                # if the defaults change, the logic here may also need to change
                if key == "from_year":
                    filter_conditions.append(f"source.date_pub >= to_timestamp('{current_value}-01-01')")
                elif key == "to_year":
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
            mode = "chat"  # mode is either 'chat' for a chat wit memeory or 'indiv' to return one response per doc
        merge = mode == "indiv"
        limit = 10

        if mode not in possible_modes:
            raise Exception('Mode must be "chat" or "indiv"')

        llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini")
        indiv_qa_chain = create_stuff_documents_chain(llm, basic_question_prompt)
        chat_qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        retriever = CustomRetriever()
        # credit: https://medium.com/@eric_vaillancourt/mastering-langchain-rag-integrating-chat-history-part-2-4c80eae11b43
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
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
            <h2>🧠 Nesta Brain</h2><br/>This is an experimental prototype of a chatbot that "knows" a lot of about Nesta.
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
                on_change=filter_flag,
            )
            to_year = st.number_input(
                label="To year",
                min_value=from_year,
                max_value=CURRENT_YEAR,
                key="to_year",
                value=WIDGET_DEFAULTS["to_year"],
                on_change=filter_flag,
            )
            include_people_options = ("Yes", "No")
            include_people = st.radio(
                "Include people pages",
                include_people_options,
                key="include_people",
                index=include_people_options.index(WIDGET_DEFAULTS["include_people"]),
                on_change=filter_flag,
            )
            mission_options = ("A fairer start", "A healthy life", "A sustainable future", None)
            mission = st.radio(
                "Mission-specific content",
                mission_options,
                key="mission",
                index=mission_options.index(WIDGET_DEFAULTS["mission"]),
                on_change=filter_flag,
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
            with st.chat_message("assistant"), st.empty():

                retriever.filter_condition = (
                    filter_conditions()
                )  # this is not ideal syntax, but kwargs to chain.invoke are not passed on to the retriever

                if mode == "indiv":
                    if input:
                        with st.spinner("Fetching documents ..."):
                            chunks = retriever.invoke(input, limit=limit, merge=merge)
                else:
                    chunks = []  # if mode == 'chat', retrieval is already part of the chain

                if mode == "chat" or (mode == "indiv" and chunks):
                    with st.spinner("Sending retrieved chunks to LLM with query ..."):
                        responses = respond(rag_chain if mode == "chat" else indiv_qa_chain, chunks, input, mode)

                if responses:
                    for response in responses:
                        st.markdown(response.as_html(), unsafe_allow_html=True)
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
