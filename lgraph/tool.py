from __future__ import annotations

import re

from enum import Enum

from pydantic import BaseModel
from pydantic import Field


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


class OfficeTemplate(BaseModel):
    """
    A class for describing templates for proposals, project updates, etc.

    Note: this class is not used for information extraction but for defining possible templates
    """

    UID: str = (
        Field(
            ...,
            description="A unique identifier",
        ),
    )
    title: str = (
        Field(
            ...,
            description="The template title",
        ),
    )
    purpose: str = (
        Field(
            ...,
            description="What the template is for",
        ),
    )
    location: str

    def __repr__(self) -> str:
        """Self-explanatory"""
        format = "{UID}: {title}. {purpose}"
        return format.format(**{k: getattr(self, k) for k in self.__class__.dict() if k in format})


class PossibleTemplate(Enum):
    """Specifies which office templates exist"""

    PROJ = OfficeTemplate(
        UID="PROJ",
        title="Project Proposal Template",
        purpose="Help staff write proposals at the Opportunity and Scoping phases",
        location="https://docs.google.com/document/d/1cOv2vXcIPWQmRPeVDB-JMz8rS1CUIkyUeCFR-_7HyqY",
    )
