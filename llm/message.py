from __future__ import annotations

import importlib
import logging
import re

from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

import markdown

from config import DEBUG_MODE
from config import PROJECT
from dsp_nesta_brain import logger
from langchain.docstore.document import Document as LangchainDocument
from langchain_core.messages import AIMessage
from retrieval.db.schema.nesta_brain import Chunk as NestaBrainChunk
from retrieval.db.schema.nesta_brain import MissionProject
from retrieval.db.schema.policy_atlas import Activity


if PROJECT == "NESTA_BRAIN":
    Chunk = NestaBrainChunk
    SCHEMA_MODULE = importlib.import_module("retrieval.db.schema.nesta_brain")

elif PROJECT == "POLICY_ATLAS":
    Chunk = Activity
    SCHEMA_MODULE = importlib.import_module("retrieval.db.schema.policy_atlas")


class Reference(LangchainDocument):
    """Class to represent retrieved documents for the purposes of presentation, and for making in-line citations easier.

    index:  the initial index of the reference representing its position in the list of retrieved documents

            (starting at 1, not 0); this is the index which the LLM 'sees' for the purposes of inline citations

    reset_index: the final index of the reference for presentational purposes, given that references are reordered so that

                 those which are cited appear before those which weren't cited

                 (see `reset_reference_indices` method of CustomAIMessage)

    cited: whether the document was used as an in-line citation or not
    """

    index: int
    reset_index: Optional[int] = None
    cited: bool = False
    chunk_class: Type = Chunk

    def __init__(self, chunk: LangchainDocument, index: int) -> None:

        if chunk.metadata.get("schema"):
            chunk_class = getattr(SCHEMA_MODULE, chunk.metadata.get("schema"))
        else:
            chunk_class = Chunk

        super().__init__(
            page_content=chunk.page_content, metadata=chunk_class.reference_metadata(**chunk.metadata), index=index
        )

        self.chunk_class = chunk_class
        # oddly, this syntax raised a pydantic error: self.chunk_class = getattr(SCHEMA_MODULE,chunk.metadata.get('schema'))

    @property
    def is_internal_policy_document(self) -> bool:
        """Return whether the document is an internal policy document"""
        return self.metadata.get("drive_type") == "policy" and "https://drive.google.com/file/d" in self.metadata.get(
            "location"
        )

    def as_html(self, reset_index: bool = False) -> str:
        """Return reference metadata as an anchor element (indexed)"""

        date_format = None
        if self.is_internal_policy_document:
            date_format = "(%B %Y)"

        reference_html_format = self.chunk_class.reference_html_format(add_date=bool(date_format) or DEBUG_MODE)

        index = self.reset_index if reset_index is not None else self.index
        if DEBUG_MODE:
            # a warning that DEBUG_MODE is on is displayed via the UI
            reference_html_format = reference_html_format.replace("</a>", " {contentType} {missions}</a>")

        # quick hack for the policy atlas
        if PROJECT == "POLICY_ATLAS":
            self.metadata["reporting_org_narrative"] = str(self.metadata["reporting_org_narrative"]).replace(
                "UK - Foreign, Commonwealth Development Office (FCDO)", "FCDO"
            )

        metadata = self.metadata.copy()
        if date_format:
            metadata["date_pub"] = datetime.strftime(metadata["date_pub"], date_format)
        return reference_html_format.format(index=index, **metadata)

    def as_superscript(self, reset_index: bool = False) -> str:
        """Return index as a (usually) clickable link within a superscript, suitable for inline citations"""
        superscript_html_format = '<sup><a href="{url}">{index}</a></sup>'
        index = self.reset_index if reset_index is not None else self.index
        if PROJECT == "POLICY_ATLAS":
            return superscript_html_format.format(url=self.metadata.get("url"), index=index)
        return superscript_html_format.format(url=self.metadata.get("location"), index=index)


class CustomAIMessage(AIMessage):
    """An extension of LangChain's AIMessage just to make printing and writing responses to streamlit easier"""

    role: str = "assistant"
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
        """Return string representation"""
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
        if not self.citations_have_been_flagged:
            self.flag_citations()
        return [reference for reference in self.references if reference.cited]

    #  @property
    # def p_element(self) -> str:
    #    """Return content as an HTML paragraph"""
    #   return f"<p>{self.content_with_superscript_citations}</p>"

    @property
    def references_(self) -> str:
        """Return formatted reference list"""
        cited = [
            reference.as_html(reset_index=True) for reference in self.cited_references
        ]  # self.cited_references could also include projects – let them be listed here if cited
        not_cited = [
            reference.as_html(reset_index=True)
            for reference in self.uncited_references
            if reference.chunk_class is Chunk
        ]  # self.uncited_references could also include projects
        actual_references = "<br><em>Cited references:</em><br>" + "<br>".join(cited) if cited else ""
        the_rest = (
            f"<br><br><em>{'May be useful' if cited else 'May be useful'}:</em><br>" + "<br>".join(not_cited)
            if not_cited
            else ""
        )
        projects = [
            reference.as_html(reset_index=True)
            for reference in self.uncited_references
            if reference.chunk_class is MissionProject
        ]
        projects = "<br><em>Potentially relevant projects:</em><br>" + "<br>".join(projects) if projects else ""
        return actual_references + the_rest + projects

    @property
    def content_as_html(self) -> str:
        """Convert the content into HTML"""
        html = markdown.markdown(self.content)
        # in lists of bullet points with blockquotes, this tends to render each bullet point as its own ordered list,
        # losing the numbering
        html = re.sub(r"</ol>\n<blockquote>", "<blockquote>", html, re.M)
        html = re.sub(r"</blockquote>\n<ol>", "</blockquote>", html, re.M)
        return html

    @property
    def content_with_superscript_citations(self) -> str:
        """
        Return content converting all citations in square brackets to a clickable superscript

        Note: we may encounter problems if for some reason numbers within square brackets appear in the content
        because they are part of the answer
        """

        if not self.reference_indices_have_been_reset:
            self.reset_reference_indices()

        content = self.content_as_html
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
        if not self.citations_have_been_flagged:
            self.flag_citations()
        return [reference for reference in self.references if not reference.cited]

    @property
    def citations_have_been_flagged(self) -> bool:
        """
        Test whether any of the references have cited property set to True.

        Also return True if there are no citations
        """

        return any(reference.cited for reference in self.references) or not self.citations_in_content

    @property
    def reference_indices_have_been_reset(self) -> bool:
        """Test whether the references' reset_index has been set"""
        return all(reference.reset_index is not None for reference in self.references)

    def as_html(self) -> str:
        """Convert the response into HTML"""
        format = '<div class="response">{main_content}{references}</div>'
        return format.format(
            main_content=self.content_with_superscript_citations,
            references=self.references_ if self.references else "",
        )

    def flag_citations(self) -> None:
        """Flag references which have been cited"""
        for citation in self.citations_in_content:
            citation_index = int(citation[1:-1])
            reference = self.references[citation_index - 1]
            reference.cited = True

    def reset_reference_indices(self) -> None:
        """Reset how the reference numbering will appear if references are split into cited and uncited sources"""
        for i, reference in enumerate(self.cited_references + self.uncited_references):
            reference.reset_index = i + 1


class InterimAIMessage(AIMessage):
    """
    AI Messages derived during intermediate steps (like recontextualisation) that are not
    supposed to be part of the chat history
    """  # noqa

    pass


class EditableAIMessage(CustomAIMessage):
    """A Custom AI Message carrying a human-edited version of its content"""

    edited: Optional[str] = None

    @property
    def p_element_edited(self) -> str:
        """Return content as an HTML paragraph"""
        if self.edited:
            return f"<p>{self.edited}</p>"
        else:
            logger.warning("No edited content available – returning original content")
            return self.p_element

    def as_html(self, edited: bool = False) -> str:
        """Convert the response into HTML"""
        if edited:
            return f'<div class="response"><i>(edited)</i> {self.p_element_edited}</div>'
        else:
            return super().as_html()

    def edit(self, edited_text: str) -> None:
        """Edit the content of the message"""
        self.edited = edited_text
