from __future__ import annotations

import re

from typing import TYPE_CHECKING
from typing import List

import openparse

from openparse.schemas import TableElement
from scraping.pdf.base import BasePDF


if TYPE_CHECKING:
    from pydantic import BaseModel


class OpenParsePDF(BasePDF):
    """
    Represents a scraped PDF document

    This is intended for short, simple documents which may contain tables (like policy documents from Google Drive)
    but which do not contain unwanted elements like headers, footers, etc.
    """

    location: str
    elements: List[BaseModel]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parse()

    @staticmethod
    def parse_table(table: TableElement) -> str:
        """Parse a table element into a readable string"""
        text = table.text
        text = re.sub(r"(\w) \| (\w)", "\\1: \\2", text)
        text = re.sub(r"\|[ -]+", "", text)
        text = re.sub(r"\|\n", "\n", text)
        table.text = text
        return table

    @property
    def text(self) -> str:
        """Return the text of the document"""
        return "\n".join([element.text for element in self.elements])

    def parse(self) -> None:
        """Parse the document"""
        parser = openparse.DocumentParser(
            table_args={"parsing_algorithm": "pymupdf", "table_output_format": "markdown"}
        )

        parsed = parser.parse(self.location)
        self.elements = sum([list(node.elements) for node in parsed.nodes], [])
        for i, element in enumerate(self.elements):
            if isinstance(element, TableElement):
                self.elements[i] = self.parse_table(element)
