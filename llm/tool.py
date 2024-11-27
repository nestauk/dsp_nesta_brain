import os

from operator import itemgetter
from typing import TYPE_CHECKING
from typing import Callable
from typing import List

from dotenv import load_dotenv  # noqa
from dsp_nesta_brain import logger  # noqa
from langchain.chains import LLMChain  # noqa
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI  # noqa
from llm.prompt import basic_question_prompt  # noqa
from llm.prompt import contextualize_q_prompt  # noqa
from llm.prompt import qa_prompt  # noqa
from pydantic import BaseModel
from pydantic import Field
from retrieval.retrieve import CustomRetriever  # noqa


if TYPE_CHECKING:
    from langchain.prompts import PromptTemplate
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.runnables.base import Runnable


# see https://python.langchain.com/v0.1/docs/use_cases/question_answering/citations/


class Citation(BaseModel):
    """A class for capturing citations"""

    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class quoted_answer(BaseModel):
    """
    A class for capturing the LLM response and list of citations

    The proper CamelCase naming convention for classes is intentionally ignored to make the syntax below more readable.
    """

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(..., description="Citations from the given sources that justify the answer.")


def llm_response(chain: LLMChain, question: str, **kwargs) -> str:
    """
    Get synchronous LLM response from chain

    This is just a simplified version of the function of the same name in app.py, for testing
    """
    input = {"input": question, "chat_history": []}
    response = chain.invoke(input, **kwargs)
    return response


def rag_chain_with_citation_tool(
    retriever: BaseRetriever, llm: BaseChatModel, prompt: PromptTemplate, chat_history_func: Callable
) -> Runnable:
    """Return a RAG retrieval chain with incorporating a tool for capturing citations"""
    llm_with_tool = llm.bind_tools(
        [quoted_answer],
        tool_choice="quoted_answer",
    )
    output_parser = JsonOutputKeyToolsParser(key_name="quoted_answer", first_tool_only=True)

    answer = prompt | llm_with_tool | output_parser
    chain = (
        RunnableParallel(input=RunnablePassthrough(), docs={}, chat_history=chat_history_func)
        .assign(
            context=itemgetter("docs")
        )  # the langchain example documentation in https://python.langchain.com/v0.1/docs/use_cases/question_answering/citations/
        # uses format_docs_with_id here – this is not needed because chunk enumeration is already happening within the retriever
        # (as long as enumerate_=True in CustomRetriever.chunks_to_docs)
        .assign(quoted_answer=answer)
        .pick(["quoted_answer", "docs"])
    )

    return create_retrieval_chain(retriever, chain)


if __name__ == "__main__":

    # fot testing

    load_dotenv()

    llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini")

    retriever = CustomRetriever(merge=True)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    rag_chain = rag_chain_with_citation_tool(history_aware_retriever, llm, qa_prompt, lambda *args: [])

    resp = llm_response(rag_chain, "What work has Nesta done on climate adaptation")

# print(resp.keys(), "\n\n")
# print(resp)
