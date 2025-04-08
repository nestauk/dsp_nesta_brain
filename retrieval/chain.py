from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import Optional

from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from lgraph.graph import create_retrieval_graph
from lgraph.graph import graph_options_type
from llm.llm import default_llm as llm
from llm.message import InterimAIMessage
from llm.prompt import contextualize_q_prompt
from retrieval.retrieve import CustomRetriever


if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelLike
    from langchain_core.prompts import BasePromptTemplate
    from langchain_core.retrievers import RetrieverLike
    from langchain_core.retrievers import RetrieverOutputLike
    from langchain_core.runnables import Runnable
    from retrieval.retrieve import RetrieverInput


def create_history_aware_retriever(
    llm: LanguageModelLike,
    retriever: RetrieverLike,
    contextualisation_prompt: BasePromptTemplate,
) -> RetrieverOutputLike:
    """Create a chain that takes conversation history and returns documents.

    Modified version of:
    https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/history_aware_retriever.py
    As with create_retriever_chain, the modification is to allow a dict containing the query, filter conditions,
    and possibly other parameters to be passed through to _get_relevant_documents
    """

    def reform_as_retriever_input(dict_: Dict) -> RetrieverInput:
        # turn the output of the contextualisation chain into retriever input
        original_input = dict_["input"]
        contextualisation_response = InterimAIMessage(dict_["contextualisation"])
        reformed_input = original_input
        reformed_input["messages"].append(contextualisation_response)
        return reformed_input

    parser = lambda ai_message: ai_message.content  # noqa
    contextualisation_chain = contextualisation_prompt | llm | parser
    contextualisation_chain = RunnableParallel(
        contextualisation=contextualisation_chain, input=RunnablePassthrough()
    ) | (lambda x: reform_as_retriever_input(x))

    retrieve_documents: RetrieverOutputLike = RunnableBranch(
        (
            lambda x: len(x.get("messages") or []) == 1,
            # if the chat history is only one message long, then it just includes the user's first input
            # just pass input directly to the retriever
            retriever,
        ),
        # If there is a chat history involving AI responses, then we pass inputs to the contextualisation_chain, then to retriever
        contextualisation_chain | retriever,
    ).with_config(run_name="chat_retriever_chain")
    return retrieve_documents


def retriever(use_graph: Optional[graph_options_type] = None) -> Runnable:
    """Return a CustomRetriever with the option of chaining it with a graph in order to make retrieval more sophisticated"""
    retriever_ = CustomRetriever()

    if use_graph in ["retrieval"]:

        # the output of the graph is a list of dicts in this format:
        # List[{'node_1_name':dict representing state returned by node 1} .. {'node_n_name': state returned by node n}]
        # the intermediate steps in the chain here transform this graph output into a useful input for retriever_,
        # that is a single dict representing the final graph state
        # This final state should have the same keys as RetrieverInput

        retriever_ = (
            create_retrieval_graph()
            | (
                lambda x: x[-1] if type(x) is list else x
            )  # The output of this is this dict: {'node_n_name': state returned by node n}
            | (lambda d: d.popitem()[1])  # This output of this is a dict representing the state returned by node n
            | retriever_
        )

    return retriever_


# credit: https://medium.com/@eric_vaillancourt/mastering-langchain-rag-integrating-chat-history-part-2-4c80eae11b43
def history_aware_retriever(**kwargs) -> Runnable:
    """Return a history aware retriever while passing kwargs through to retriever"""
    return create_history_aware_retriever(llm, retriever(**kwargs), contextualize_q_prompt)
