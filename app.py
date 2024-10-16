import asyncio
import logging
import os
import sys

from typing import List
from typing import Optional
from typing import Union

from langchain.docstore.document import Document as LangchainDocument


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
from langchain.chains.combine_documents import create_stuff_documents_chain  # noqa
from langchain_openai import ChatOpenAI  # noqa
from llm.prompt import question_prompt  # noqa
from retrieval.retrieve import retrieve  # noqa


class Response:
    """A class just to make things like printing and writing to streamlit easier"""

    text: str
    is_summary: bool = False
    chunks: List[LangchainDocument]
    index: Optional[int] = None

    def __init__(
        self,
        text: str,
        chunks: Union[LangchainDocument, List[LangchainDocument]],
        index: Optional[int] = None,
        is_summary: bool = False,
    ) -> None:
        text = text.replace("ANSWER: ", "")
        self.text = text
        if isinstance(chunks, LangchainDocument):
            chunks = [chunks]
        self.chunks = chunks
        self.is_summary = is_summary
        self.index = index

    def __repr__(self) -> str:
        """Self-explanatory"""
        string = "\n--------------\n" + self.text
        if not self.is_summary:
            string += f'\n{self.chunks[0].page_content}\n{self.chunks[0].metadata["location"]}'
        string += "\n--------------\n\n"
        return string

    @property
    def a_element(self) -> str:
        """Return hyperlink to source document"""
        if self.is_summary:
            return ""
        else:
            return f'<a href="{self.chunks[0].metadata["location"]}">{self.chunks[0].metadata["title"]}</a>'

    @property
    def p_element(self) -> str:
        """Return response text as an HTML paragraph"""
        return f'<p>{"<b>SUMMARY:</b> " if self.is_summary else (f"({self.index}) " if self.index else "")}{self.text}</p>'

    def as_html(self) -> str:
        """Convert the response into HTML"""
        css_class = "response " + ("summary" if self.is_summary else "indiv")
        return f'<div class="{css_class}">{self.p_element}{self.a_element}</div>'


def llm_response(chain: LLMChain, docs: List[LangchainDocument], question: str, **kwargs) -> str:
    """Get synchronous LLM response from chain"""
    input = {"context": docs, "question": question}
    return chain.invoke(input, **kwargs)


async def async_llm_response(chain: LLMChain, docs: List[LangchainDocument], question: str, **kwargs) -> str:
    """Get asynchronous LLM response from chain"""
    input = {"context": docs, "question": question}
    return await chain.ainvoke(input, **kwargs)


async def individual_responses(chain: LLMChain, docs: List[LangchainDocument], question: str, **kwargs) -> List[str]:
    """Get asynchronous LLM responses for a number of documents/chunks from chain"""
    tasks = [asyncio.create_task(async_llm_response(chain, [doc], question, **kwargs)) for doc in docs]
    responses = await asyncio.gather(*tasks)
    return responses


def respond(
    chain: LLMChain,
    docs: List[LangchainDocument],
    question: str,
    summary_response: bool = True,
    individual_responses_: bool = False,
) -> List[Response]:
    """Get individual and/or summary responses from chain and convert them into Response objects"""

    responses = []
    if individual_responses_:
        responses_ = asyncio.run(individual_responses(chain, docs, question))
        responses_ = [(response, docs[i]) for i, response in enumerate(responses_) if response != "NULL"]
        responses += [Response(response, doc, index=i + 1) for i, (response, doc) in enumerate(responses_)]
    if summary_response:
        response = llm_response(chain, docs, question)
        if response != "NULL":
            response = Response(response, docs, is_summary=True)
            responses.append(response)

    for response in enumerate(responses):
        logger.info(response)

    return responses


def is_html(string: str) -> bool:
    """Test whether a string is HTML"""
    # credit: https://stackoverflow.com/questions/24856035/how-to-detect-with-python-if-the-string-contains-html-code
    return lxml.html.fromstring(string).find(".//*") is not None


if __name__ == "__main__":

    load_dotenv()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # settings
    # retrieval settings
    summary_response = True
    individual_responses_ = True
    merge = True
    limit = 10

    query = "climate change projects"

    if not summary_response and not individual_responses_:
        raise Exception("One or both of summary_response or individual_responses_ must be True")

    # memory = ConversationBufferMemory(memory_key="chat_history",input_key="question")
    llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo")
    chain = create_stuff_documents_chain(llm, question_prompt)

    st.set_page_config(layout="wide")
    st.markdown(
        """
    <style>
        body {
            font-family: Helvetica, Sans-Serif;
        }

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
        "<h2>Demo</h2>",
        unsafe_allow_html=True,
    )

    # Store session variables
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if is_html(message["content"]):
                st.markdown(message["html"], unsafe_allow_html=True)
            else:
                st.write(message["content"])

    # User-provided input
    if input := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": input})
        with st.chat_message("user"):
            st.write(input)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):

            if input:
                with st.spinner("Fetching documents ..."):
                    chunks = retrieve(query, limit=limit, merge=merge)

            responses = []
            if chunks:
                with st.spinner("Sending retrieved chunks to LLM with query ..."):
                    responses = respond(
                        chain,
                        chunks,
                        input,
                        summary_response=summary_response,
                        individual_responses_=individual_responses_,
                    )

            if responses:
                for response in responses:
                    st.markdown(response.as_html(), unsafe_allow_html=True)
                    message = {"role": "assistant", "content": response.as_html()}
                    st.session_state.messages.append(message)
            else:
                st.write("I was not able to answer that question")
