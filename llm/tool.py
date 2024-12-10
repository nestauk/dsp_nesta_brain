from typing import TYPE_CHECKING
from typing import List

from pydantic import BaseModel
from pydantic import Field


if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

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


def llm_response(chain: Runnable, question: str, **kwargs) -> str:
    """
    Get synchronous LLM response from chain

    This is just a simplified version of the function of the same name in app.py, for testing
    """
    input = {"input": question, "chat_history": []}
    response = chain.invoke(input, **kwargs)
    return response
