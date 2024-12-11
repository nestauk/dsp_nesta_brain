from __future__ import annotations

import re

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


class date_range(BaseModel):
    """
    A class for capturing date ranges

    I tested this in decide_if_need_time_constraint but the formats of the dates returned was not always reliable.
    It is safer to use year_range.
    """

    start_date: int = Field(
        ...,
        description="The first date in a range.",
    )
    end_date: str = Field(
        ...,
        description="The last date in a range.",
    )

    def get_date_string(self, attr_name: str) -> str:
        """Check date string is in the right format and return"""
        date_string = str(getattr(self, attr_name))  # getting an int sometimes
        if date_string and re.match("[0-9]{8}", date_string):
            date_string = re.sub("([0-9]{4})([0-9]{2})([0-9]{2})", "\\1-\\2-\\3", date_string)
        return date_string

    def to_filter_condition(self) -> str:
        """Compile filter condition from date range"""
        format_ = "source.date_pub {comparison_operator} to_timestamp('{date_string}')"
        conditions = [
            format_.format(
                comparison_operator=">=" if attr == "start_date" else "<=", date_string=self.get_date_string(attr)
            )
            for attr in ["start_date", "end_date"]
        ]
        return " and ".join(conditions)


class year_range(BaseModel):
    """A class for capturing year ranges"""

    start_year: int = Field(
        ...,
        description="The first year in a range.",
    )
    end_year: str = Field(
        ...,
        description="The last year in a range.",
    )

    def get_date_string(self, attr_name: str) -> str:
        """Return appropriate date string for filter condition for start_year and end_year"""
        if attr_name == "start_year":
            return f"{self.start_year}-01-01"
        else:
            return f"{self.end_year}-12-31"

    def to_filter_condition(self) -> str:
        """Compile filter condition from year range"""
        format_ = "source.date_pub {comparison_operator} to_timestamp('{date_string}')"
        conditions = [
            format_.format(
                comparison_operator=">=" if attr == "start_year" else "<=", date_string=self.get_date_string(attr)
            )
            for attr in ["start_year", "end_year"]
        ]
        return " and ".join(conditions)


def llm_response(chain: Runnable, question: str, **kwargs) -> str:
    """
    Get synchronous LLM response from chain

    This is just a simplified version of the function of the same name in app.py, for testing
    """
    input = {"input": question, "chat_history": []}
    response = chain.invoke(input, **kwargs)
    return response
