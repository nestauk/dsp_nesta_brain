from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from dsp_nesta_brain import logger
from langchain.chains.combine_documents.base import DEFAULT_DOCUMENT_PROMPT
from langchain.chains.combine_documents.base import DEFAULT_DOCUMENT_SEPARATOR
from langchain.chains.combine_documents.base import DOCUMENTS_KEY
from langchain.chains.combine_documents.base import _validate_prompt
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from lgraph.graph import create_chat_graph
from lgraph.graph import create_combined_graph
from lgraph.graph import graph_options_type
from llm.llm import default_llm as llm
from llm.message import InterimAIMessage
from llm.prompt import qa_prompt
from llm.tool import quoted_answer
from retrieval.chain import history_aware_retriever


if TYPE_CHECKING:
    from langchain.docstore.document import Document as LangchainDocument
    from langchain.prompts import PromptTemplate
    from langchain_core.language_models import LanguageModelLike
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.output_parsers import BaseOutputParser
    from langchain_core.prompts import BasePromptTemplate
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.runnables import Runnable
    from retrieval.retrieve import RetrieverInput


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


def filter_messages(input: RetrieverInput) -> RetrieverInput:
    """Filter out InterimAIMessages as ther shouldn't be sent to the LLM"""
    input["messages"] = [message for message in input["messages"] if not isinstance(message, InterimAIMessage)]
    return input


def filter_context(docs: List[LangchainDocument]) -> List[LangchainDocument]:
    """Filter out any docs which are flagged as not to be used for context in their metadata"""
    return [
        doc for doc in docs if doc.metadata.get("use_as_context") != False  # noqa
    ]  # Use != False rather than truthiness of doc.metadata.get("use_as_context") as there may be None values


def create_stuff_documents_chain(
    llm: LanguageModelLike,
    prompt: BasePromptTemplate,
    *,
    output_parser: Optional[BaseOutputParser] = None,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = DEFAULT_DOCUMENT_SEPARATOR,
    document_variable_name: str = DOCUMENTS_KEY,
) -> Runnable[Dict[str, Any], Any]:
    """Create a chain for passing a list of Documents to a model.

    Modified version of LangChain's create_stuff_documents_chain
    See:
    https://api.python.langchain.com/en/latest/_modules/langchain/chains/combine_documents/stuff.html#create_stuff_documents_chain

    The reason for the modification is that retriever results now may actually include some things we want to present in the UI,
    but not actually use as context. They therefore need to be filtered out here using filter_context.

    """

    _validate_prompt(prompt, document_variable_name)
    _document_prompt = document_prompt or 
    
    
    _output_parser = output_parser or StrOutputParser()

    def format_docs(inputs: dict) -> str:
        return document_separator.join(
            format_document(doc, _document_prompt) for doc in filter_context(inputs[document_variable_name])
        )

    return (
        RunnablePassthrough.assign(**{document_variable_name: format_docs}).with_config(run_name="format_inputs")
        | prompt
        | llm
        | _output_parser
    ).with_config(run_name="stuff_documents_chain")


def create_retrieval_chain(
    retriever: BaseRetriever,
    combine_docs_chain: Runnable[Dict[str, Any], str],
) -> Runnable:
    """
    Create retrieval chain that retrieves documents and then passes them on.

    Lightly modified version of:
    https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/retrieval.py
    The modification is to allow a dict containing the query, filter conditions, and possibly other
    parameters to be passed through to _get_relevant_documents
    """

    combine_docs_chain = (lambda x: filter_messages(x)) | combine_docs_chain

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

    logger.warning(
        "rag_chain_with_citation_tool is deprecated. Amongst the features it does not incorporate are:\n* Context filtering"
    )

    llm_with_tool = llm.bind_tools(
        [quoted_answer],
        tool_choice="quoted_answer",
    )
    output_parser = JsonOutputKeyToolsParser(key_name="quoted_answer", first_tool_only=True)

    answer = prompt | llm_with_tool | output_parser
    chain = (lambda x: filter_messages(x)) | (
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
