from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnableParallel
from lgraph.graph import create_retrieval_graph
from llm.llm import default_llm as llm
from llm.prompt import contextualize_q_prompt
from retrieval.retrieve import CustomRetriever


if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelLike
    from langchain_core.prompts import BasePromptTemplate
    from langchain_core.retrievers import RetrieverLike
    from langchain_core.retrievers import RetrieverOutputLike
    from langchain_core.runnables import Runnable


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


def retriever(use_langgraph: bool = False) -> Runnable:
    """Return a CustomRetriever with the option of chaining it with a graph in order to make retrieval more sophisticated"""
    retriever_ = CustomRetriever()
    if use_langgraph:

        retriever_ = (
        # Create a retrieval graph, which returns a list of dicts with node outputs, such as:
        # List[{'node_1_name':dict representing state returned by node 1} .. 
        # {'node_n_name': state returned by node n}]
        create_retrieval_graph()
        # Extract the last element
        | (lambda x: x[-1] if type(x) is list else x)
        # Extract the value from a dictionary by removing its only key-value pair, and returning only the value
        | (lambda d: d.popitem()[1])
        # Pass the refined output into the CustomRetriever for final retrieval processing
        | retriever_
    )
    
    return retriever_


# credit: https://medium.com/@eric_vaillancourt/mastering-langchain-rag-integrating-chat-history-part-2-4c80eae11b43
def history_aware_retriever(**kwargs) -> Runnable:
    """Return a history aware retriever while passing kwargs through to retriever"""
    return create_history_aware_retriever(llm, retriever(**kwargs), contextualize_q_prompt)
