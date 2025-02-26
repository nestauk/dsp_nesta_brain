from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Optional

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from lgraph.graph import create_chat_graph
from lgraph.graph import create_combined_graph
from lgraph.graph import graph_options_type
from llm.llm import default_llm as llm
from llm.prompt import qa_prompt
from llm.tool import quoted_answer
from retrieval.chain import history_aware_retriever


if TYPE_CHECKING:
    from langchain.prompts import PromptTemplate
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.prompts import BasePromptTemplate
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.runnables import Runnable


def get_graph_or_rag_chain(
    use_tool_for_citations: bool = False,
    use_graph: Optional[graph_options_type] = None,
    return_stream_nodes: bool = False,
    **kwargs,
) -> Runnable:
    """Return a suitable graph or RAG chain depending on the arguments"""

    if use_tool_for_citations:
        raise Exception("use_tool_for_citations may no longer work – need to check")

    if use_graph == "chat":
        return create_chat_graph(
            use_tool_for_citations=use_tool_for_citations, return_stream_nodes=return_stream_nodes, **kwargs
        )
    elif use_graph == "combined":
        return create_combined_graph(
            use_tool_for_citations=use_tool_for_citations, return_stream_nodes=return_stream_nodes, **kwargs
        )

    elif use_tool_for_citations:
        runnable = history_aware_rag_chain_with_citation_tool(use_graph=use_graph, **kwargs)
    else:
        runnable = history_aware_rag_chain(use_graph=use_graph, **kwargs)

    if return_stream_nodes:
        return runnable, []  # stream_nodes are only relevant for graphs
    else:
        return runnable


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


def rag_chain_with_citation_tool(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    prompt: PromptTemplate,
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

    # answer = create_stuff_documents_chain(llm_with_tool,prompt,output_parser=output_parser)
    answer = prompt | llm_with_tool | output_parser
    chain = (
        RunnableParallel(
            input=(lambda x: x["messages"][-1]),
            context=(lambda x: x["context"]),
            chat_history=(lambda x: x["messages"][0:-1]),
        )
        .assign(quoted_answer=answer)
        .pick(["quoted_answer"])
    )

    return create_retrieval_chain(retriever, chain)


def history_aware_rag_chain(prompt: BasePromptTemplate = qa_prompt, **kwargs) -> Runnable:
    """Return a history aware RAG chain while passing kwargs through to history_aware_retriever"""
    chat_qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(history_aware_retriever(**kwargs), chat_qa_chain)


def history_aware_rag_chain_with_citation_tool(prompt: BasePromptTemplate = qa_prompt, **kwargs) -> Runnable:
    """Return a history aware RAG chain with citation tool while passing kwargs through to history_aware_retriever"""
    return rag_chain_with_citation_tool(history_aware_retriever(**kwargs), llm, prompt)
