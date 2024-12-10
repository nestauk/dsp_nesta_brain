from __future__ import annotations

import os

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langgraph.langgraph_ import graph
from llm.prompt import contextualize_q_prompt
from llm.prompt import qa_prompt
from llm.tool import quoted_answer
from retrieval.retrieve import CustomRetriever


if TYPE_CHECKING:
    from langchain.prompts import PromptTemplate
    from langchain_core.language_models import LanguageModelLike
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.prompts import BasePromptTemplate
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.retrievers import RetrieverLike
    from langchain_core.retrievers import RetrieverOutputLike
    from langchain_core.runnables import Runnable


def create_retrieval_chain(
    retriever: BaseRetriever,
    combine_docs_chain: Runnable[Dict[str, Any], str],
) -> Runnable:
    """
    Create retrieval chain that retrieves documents and then passes them on.

    Lightly modified version of:
    https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/retrieval.py
    The modification is to allow a dict containing the query, filter conditions, and possibly other paramters to be passed through
    to _get_relevant_documents
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=retriever.with_config(run_name="retrieve_documents"),
        ).assign(answer=combine_docs_chain)
    ).with_config(run_name="retrieval_chain")

    return retrieval_chain


def create_history_aware_retriever(
    llm: LanguageModelLike,
    retriever: RetrieverLike,
    prompt: BasePromptTemplate,
) -> RetrieverOutputLike:
    """Create a chain that takes conversation history and returns documents.

    Modified version of:
    https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/history_aware_retriever.py
    As with create_retriever_chain, the modification is to allow a dict containing the query, filter conditions,
    and possibly other parameters to be passed through to _get_relevant_documents
    """

    parser = lambda ai_message: ai_message.content  # noqa
    recontextualisation_chain = prompt | llm | parser
    recontextualisation_chain = RunnableParallel(
        input=recontextualisation_chain,
        filter_condition=lambda x: x.get("filter_condition"),
        merge=lambda x: x.get("merge") or False,
    )
    # unlike in the original version of create_history_aware_retriever, we want filter_condition and merge
    # to be passed through to the retriever

    retrieve_documents: RetrieverOutputLike = RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("chat_history", False),
            # If no chat history, then we just pass input to retriever
            retriever,
        ),
        # If chat history, then we pass inputs to LLM chain, then to retriever
        recontextualisation_chain | retriever,
    ).with_config(run_name="chat_retriever_chain")
    return retrieve_documents


def rag_chain_with_citation_tool(
    retriever: BaseRetriever, llm: BaseChatModel, prompt: PromptTemplate, chat_history_func: Callable
) -> Runnable:
    """Return a RAG retrieval chain with incorporating a tool for capturing citations"""

    # the langchain example this is based on (see https://python.langchain.com/v0.1/docs/use_cases/question_answering/citations/)
    # uses format_docs_with_id here – this is not needed because chunk enumeration is already happening within the retriever
    # (as long as enumerate_=True in CustomRetriever.chunks_to_docs)

    llm_with_tool = llm.bind_tools(
        [quoted_answer],
        tool_choice="quoted_answer",
    )
    output_parser = JsonOutputKeyToolsParser(key_name="quoted_answer", first_tool_only=True)

    answer = prompt | llm_with_tool | output_parser
    chain = (
        RunnableParallel(input=RunnablePassthrough(), context=RunnablePassthrough(), chat_history=chat_history_func)
        .assign(quoted_answer=answer)
        .pick(["quoted_answer"])
    )

    return create_retrieval_chain(retriever, chain)


load_dotenv()

llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini", streaming=True)

chat_qa_chain = create_stuff_documents_chain(llm, qa_prompt)


def retriever(use_langgraph: bool = False) -> Runnable:
    """Return a CustomRetriever with the option of chaining it with a graph in order to make retrieval more sophisticated"""
    retriever_ = CustomRetriever()
    if use_langgraph:
        retriever_ = (
            graph | (lambda x: list(x.values())[0]) | retriever_
        )  # the lambda ensures the last state value is passed on to the retriever
    return retriever_


# credit: https://medium.com/@eric_vaillancourt/mastering-langchain-rag-integrating-chat-history-part-2-4c80eae11b43
def history_aware_retriever(**kwargs) -> Runnable:
    """Return a history aware retriever while passing kwargs through to retriever"""
    return create_history_aware_retriever(llm, retriever(**kwargs), contextualize_q_prompt)


def history_aware_rag_chain(**kwargs) -> Runnable:
    """Return a history aware RAG chain while passing kwargs through to history_aware_retriever"""
    return create_retrieval_chain(history_aware_retriever(**kwargs), chat_qa_chain)


def history_aware_rag_chain_with_citation_tool(chat_history_func: Callable, **kwargs) -> Runnable:
    """Return a history aware RAG chain with citation tool while passing kwargs through to history_aware_retriever"""
    return rag_chain_with_citation_tool(history_aware_retriever(**kwargs), llm, qa_prompt, chat_history_func)
