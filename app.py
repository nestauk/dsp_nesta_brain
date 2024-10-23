import asyncio
import logging
import os
import sys

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document as LangchainDocument
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables.base import Runnable


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


def unique(seq: Union[List, Tuple]) -> List:
    """Find unique elements of a sequence and retain order"""
    seen = {}
    result = []
    for item in seq:
        if item in seen:
            continue
        seen[item] = 1
        result.append(item)
    return result


class Response:
    """A class just to make things like printing and writing to streamlit easier"""

    text: str
    mode: str
    chunks: List[LangchainDocument]
    index: Optional[int] = None

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
        #   if self.is_summary:
        #      return ""
        # else:
        elements = [f'<a href="{chunk.metadata["location"]}">{chunk.metadata["title"]}</a>' for chunk in self.chunks]
        return "<br>".join(unique(elements))

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
    return chain.invoke(input, **kwargs)


async def async_llm_response(chain: LLMChain, docs: List[LangchainDocument], question: str, **kwargs) -> str:
    """Get asynchronous LLM response from chain"""
    #  print("message history",chat_history())
    #  input = {"input": question,"chat_history":chat_history()}
    input = {"context": docs, "question": question}
    return await chain.ainvoke(input, **kwargs)


async def individual_responses(chain: LLMChain, docs: List[LangchainDocument], question: str, **kwargs) -> List[str]:
    """Get asynchronous LLM responses for a number of documents/chunks from chain"""
    tasks = [asyncio.create_task(async_llm_response(chain, [doc], question, **kwargs)) for doc in docs]
    responses = await asyncio.gather(*tasks)
    return responses


def respond(chain: Runnable, docs: List[LangchainDocument], question: str, mode: str) -> List[Response]:
    """Get individual and/or summary responses from chain and convert them into Response objects"""

    responses = []

    if mode == "indiv":
        responses_ = asyncio.run(individual_responses(chain, docs, question))
        responses_ = [(response, docs[i]) for i, response in enumerate(responses_) if response != "NULL"]
        responses += [Response(response, doc, mode, index=i + 1) for i, (response, doc) in enumerate(responses_)]

    response = llm_response(chain, docs, question, mode)
    response = Response(response, docs, mode)
    if response.text != "NULL":
        responses.append(response)

    # for response in enumerate(responses):
    #    logger.info(response)

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
    possible_modes = ["chat", "indiv"]
    if sys.argv[1:] and sys.argv[1] in ["chat", "indiv"]:
        mode = sys.argv[1]
    else:
        mode = "chat"  # mode is either 'chat' for a chat wit memeory or 'indiv' to return one response per doc
    merge = mode == "indiv"
    limit = 10

    if mode not in possible_modes:
        raise Exception('Mode must be "chat" or "indiv"')

    llm = ChatOpenAI(
        temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-#4o-mini"
    )  # gpt-3.5-turbo")
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
        f"<h2>Demo (mode = '{mode}')</h2>",
        unsafe_allow_html=True,
    )

    # Store session variables
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]

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
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"), st.empty():

            if mode == "indiv":
                if input:
                    with st.spinner("Fetching documents ..."):
                        chunks = retriever.invoke(input, limit=limit, merge=merge)
            else:
                chunks = []  # if mode == 'chat', retrieval is already part of the chain

            responses = []
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
