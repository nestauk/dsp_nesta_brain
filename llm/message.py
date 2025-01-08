from __future__ import annotations

import logging
import re

from typing import Dict
from typing import List
from typing import Optional

import markdown

from dsp_nesta_brain import logger
from langchain.docstore.document import Document as LangchainDocument
from langchain_core.messages import AIMessage


class Reference(LangchainDocument):
    """Extends the LangchainDocument class to make inline citations easier"""

    index: int
    reset_index: Optional[int] = None
    cited: bool = False

    def __init__(self, chunk: LangchainDocument, index: int) -> None:
        super().__init__(page_content=chunk.page_content, metadata=chunk.metadata, index=index)

    @property
    def is_pdf(self) -> bool:
        """Test whether the underlying source document is a PDF"""
        return self.metadata["location"].lower()[-4:] == ".pdf"

    def as_html(self, reset_index: bool = False) -> str:
        """Return reference metadata as an anchor element (indexed)"""
        test_mode = False
        index = self.reset_index if reset_index else self.index
        if test_mode:
            if self.index == 1:
                logger.warning(
                    "Formatting of links for testing retrieval filtering is in use – do not use for production"
                )
            return f'<a href="{self.metadata["location"]}">[{index}] {self.metadata["title"]}{" (PDF)" if self.is_pdf else ""} {self.metadata["date_pub"]} {self.metadata["contentType"]} {self.metadata["missions"]}</a>'  # noqa
        else:
            return f'<a href="{self.metadata["location"]}">[{index}] {self.metadata["title"]}{" (PDF)" if self.is_pdf else ""}</a>'  # noqa

    def as_superscript(self, reset_index: bool = False) -> str:
        """Return index as a clickable link within a superscript, suitable for inline citations"""
        return (
            f'<sup><a href="{self.metadata["location"]}">{self.reset_index if reset_index else self.index}</a></sup>'
        )


class CustomAIMessage(AIMessage):
    """An extension of LangChain's AIMessage just to make printing and writing responses to streamlit easier"""

    references: List[Reference]

    class Config:  # noqa
        arbitrary_types_allowed = True

    def __init__(self, chain_response: Dict) -> None:

        if type(chain_response["answer"]) is str:
            content = chain_response["answer"]
        elif isinstance(
            chain_response["answer"], dict
        ):  # this will be the case if a tool has been used for citations:
            # see quoted_answer class in llm/tool.py
            # use isinstance, not type
            content = chain_response["answer"]["quoted_answer"]

        chunks = chain_response["context"]
        references = [Reference(chunk, i + 1) for i, chunk in enumerate(chunks)]

        super().__init__(content=content, references=references)

    def __repr__(self) -> str:
        """Self-explanatory"""
        string = "\n--------------\n" + self.content
        string += f'\n{self.references[0].page_content}\n{self.references[0].metadata["location"]}'
        string += "\n--------------\n\n"
        return string

    @property
    def a_elements(self) -> str:
        """Return hyperlink(s) to source document(s)"""
        return [reference.as_html() for reference in self.references]

    @property
    def citations_in_content(self) -> List[str]:
        """Return the set of citations in the content, i.e. numbers appearing in square brackets"""
        return set(re.findall(r"\[\d+\]", self.content))

    @property
    def cited_references(self) -> List[Reference]:
        """Return a list of references which are actually cited in the content"""
        return [reference for reference in self.references if reference.cited]

    @property
    def p_element(self) -> str:
        """Return content as an HTML paragraph"""
        return f"<p>{self.content_with_superscript_citations}</p>"

    @property
    def references_(self) -> str:
        """Return formatted reference list"""
        cited = [reference.as_html(reset_index=True) for reference in self.cited_references]
        not_cited = [reference.as_html(reset_index=True) for reference in self.uncited_references]
        actual_references = "<br><em>Cited references:</em><br>" + "<br>".join(cited) if cited else ""
        the_rest = (
            f"<br><em>{'May be useful' if cited else 'May be useful'}:</em><br>" + "<br>".join(not_cited)
            if not_cited
            else ""
        )
        return actual_references + the_rest

    @property
    def content_with_superscript_citations(self) -> str:
        """
        Return content converting all citations in square brackets to a clickable superscript

        Note: we may encounter problems if for some reason numbers within square brackets appear in the content
        because they are part of the answer
        """

        self.reset_reference_indices()

        content = markdown.markdown(self.content)
        N_references = len(self.references)
        for citation in self.citations_in_content:
            citation_index = int(citation[1:-1])  # remove the square brackets
            if citation_index <= N_references:  # citation indices are in the range 1:N rather than 0:(N-1)
                reference = self.references[citation_index - 1]
                superscript = reference.as_superscript(reset_index=True)
                content = content.replace(citation, superscript)
            else:
                logging.warning(f"Citation {citation} contained an index greater than the number of references")
        content = content.replace(
            "</sup><sup>", ","
        )  # where there are citations next to each other, merge them into the same superscript and separate them with commas
        return content

    @property
    def uncited_references(self) -> List[Reference]:
        """Return a list of references which are not cited in the content"""
        return [reference for reference in self.references if not reference.cited]

    def as_html(self) -> str:
        """Convert the response into HTML"""
        return f'<div class="response">{self.p_element}{self.references_}</div>'

    def reset_reference_indices(self) -> None:
        """Reset how the reference numbering will appear if references are split into cited and uncited sources"""
        for citation in self.citations_in_content:
            citation_index = int(citation[1:-1])
            reference = self.references[citation_index - 1]
            reference.cited = True

        for i, reference in enumerate(self.cited_references + self.uncited_references):
            reference.reset_index = i + 1
