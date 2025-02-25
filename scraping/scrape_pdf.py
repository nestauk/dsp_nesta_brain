from __future__ import annotations

import datetime as dt
import itertools as it
import re

from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from dsp_nesta_brain import logger
from nltk.tokenize import sent_tokenize
from unstructured.documents.elements import Element
from unstructured.documents.elements import ListItem
from unstructured.documents.elements import NarrativeText
from unstructured.documents.elements import Text
from unstructured.documents.elements import Title
from unstructured.partition.pdf import partition_pdf
from utils import first


class PDF:
    """Represents a scraped PDF document"""

    location: str
    elements: List[Element]
    pages: List[PDFPage]
    good_pages_: Optional[List[PDFPage]] = None
    sections: List[PDFSection]  # sections is not necessarily the same as the concatenated sections of pages
    good_sections: Optional[List[PDFSection]] = None
    linking_url: Optional[str] = None

    def __init__(self, location: str, linking_url: Optional[str] = None) -> None:
        self.location = location
        logger.info(f"\nReading PDF document {self.location} ...")
        self.elements = partition_pdf(self.location)
        logger.info("Converting elements to pages and sections ...")
        self.pages = self.elements_to_pages(self.elements)
        self.sections = self.elements_to_sections(self.elements)
        self.linking_url = linking_url

    @staticmethod
    def elements_to_pages(elements: List[Element]) -> List[PDFPage]:
        """Convert a list of unstructured document elements to pages"""

        pages = {}
        for element in elements:
            if element.metadata.page_number not in pages:
                pages[element.metadata.page_number] = []
            pages[element.metadata.page_number].append(element)

        pages = [PDFPage(page_number, elements) for page_number, elements in pages.items()]

        return pages

    @staticmethod
    def elements_to_sections(elements: List[Element]) -> List[PDFSection]:
        """Convert a list of unstructured document elements to sections"""

        sections = []
        current_section = None

        for element in elements:

            start_new_section = isinstance(element, Title) and not PDF.is_bad_title(element)

            if start_new_section:
                if current_section and current_section.text_elements:
                    sections.append(current_section)
                current_section = PDFSection(element, [])

            if current_section and PDF.is_good_text_element(element):
                current_section.text_elements.append(element)

        return sections

    @property
    def filtered_text(self) -> str:
        """Collate the text from good sections"""
        if self.good_sections is None:
            self.filter()
        return "\n\n".join([section.text for section in self.good_sections])

    @staticmethod
    def is_bad_text(element: Element) -> bool:
        """Test whether element contains text we definitely don’t want"""
        is_page_number = re.match(r"PG\.? \d+$", element.text)
        is_about_licensing = re.match(
            "This work is licensed under a Creative Commons|creativecommons.org/licenses", element.text
        )
        return is_page_number or is_about_licensing or PDF.is_malformed_string(element)

    @staticmethod
    def is_bad_title(element: Title) -> bool:
        """Test whether element is a title we definitely don’t want"""
        is_single_letter = re.match(r"\(?[A-Za-z]\)?\.?$", element.text)
        is_caption = re.match("Source: ", element.text)
        return is_single_letter or is_caption or PDF.is_bad_text(element)

    @staticmethod
    def is_good_text_element(element: Element) -> bool:
        """Test whether element is desirable text"""
        return isinstance(element, (NarrativeText, ListItem)) and not PDF.is_bad_text(element)

    @staticmethod
    def is_malformed_string(string_or_element: Union[str, Element]) -> bool:
        """
        Test whether text l o o k s l i k e t h i s
        Text from diagrams is sometimes rendered this way and should be excluded
        """  # noqa

        if isinstance(string_or_element, Element):
            string = string_or_element.text
        else:
            string = string_or_element
        return re.search(r"^(\S )+\S$", string.strip())  # the whole string like this indicates malformed text

    @property
    def end_section(self) -> Union[PDFSection, None]:
        """Retrieve Endnotes or References section, if one exists"""
        return first(self.sections, lambda section: section.is_end_section)

    @property
    def executive_summary(self) -> Union[PDFSection, None]:
        """Retrieve Eexecutive Summary, if one exists"""
        return first(self.sections, lambda section: section.is_executive_summary)

    @property
    def good_pages(self) -> List[PDFPage]:
        """Filter pages according to whether they are desirable, store result and return; or return a previously stored result"""
        if self.good_pages_ is None:  # only do it once
            self.good_pages_ = [page for page in self.pages if not page.is_bad_page]
        return self.good_pages_

    @property
    def is_standard_report_format(self) -> bool:
        """Test whether the document has an Executive Summary and an Endnotes/References section,
        which indicates a standard report format"""  # noqa

        return self.executive_summary and self.end_section

    def filter(self) -> None:
        """
        Filter the document content according to rules:
        1. If the document has a standard report format, exclude content before the Executive Summary and content
        after the Endnotes/References, as well as the Endnotes/References themselves
        2. Otherwise filter out problem pages (table of contents, pages with almost no content, etc.)
        Desirable content is stored as sections in self.good_sections
        """  # noqa

        logger.info("Identifying undesirable content ...")

        if self.is_standard_report_format:
            exec_summ_and_after = list(it.dropwhile(lambda section: not section.is_executive_summary, self.sections))
            sections_before_end_section = list(
                it.takewhile(lambda section: not section.is_end_section, exec_summ_and_after)
            )
            self.good_sections = sections_before_end_section
            if self.good_sections:
                logger.info(
                    "Content before the Executive Summary (exclusive) and after the Endnotes/References (inclusive) will not be ingested"  # noqa
                )

        if self.good_sections is None and len(self.good_pages) != len(self.pages):
            elements_on_good_pages = sum([page.elements for page in self.good_pages], [])
            self.good_sections = PDF.elements_to_sections(elements_on_good_pages)
            logger.info("Content on pages identified as undesirable will not be ingested")

        if self.good_sections is None:
            msg = f"I was not sure how to identify undesirable content for PDF {self.location} - the entire contents will be ingested"  # noqa
            logger.warning(msg)

    def guess_metadata(
        self,
        title_guess: Optional[Union[str, List[str]]] = None,
        date_guess: Optional[str] = None,
        cautious: bool = False,
        force_date: bool = False,
        indent: Optional[str] = "",
    ) -> Dict:
        """Guess the title and check whether the title guess and date guess (if any) are correct"""

        if isinstance(title_guess, list):
            title_guesses = title_guess
        else:
            title_guesses = [title_guess] if title_guess else []

        logger.info(indent + "Guessing metadata ...")

        if self.pages[0].is_title_page and self.pages[0].title:
            title_guesses.append(str(self.pages[0].title))
        first_page_with_title = first(
            self.pages[1:], lambda page: first(page.elements, lambda element: isinstance(element, Title))
        )
        if first_page_with_title:
            title_guesses.append(first(first_page_with_title.elements, lambda element: isinstance(element, Title)))

        title = None
        while title_guesses and not title:
            title_guess = title_guesses.pop(0)
            if title_guess:  # it might be None by mistake
                if not cautious or input(
                    indent + f'Is this the document title: "{str(title_guess)}"? (any key except enter = "yes")'
                ):
                    title = str(title_guess)
        if not title:
            title = input(indent + "Enter document title: ")

        metadata = {"title": title}

        date_pub = None
        if date_guess:
            if not cautious or input(
                indent + f'Is this the publication date: "{date_guess}"? (any key except enter = "yes")'
            ):
                date_pub = date_guess

        if date_guess or force_date:
            while not metadata.get("date_pub"):
                try:
                    metadata["date_pub"] = dt.datetime.strptime(date_pub, "%Y-%m-%d")
                except Exception:
                    date_pub = input(indent + "Enter publication date (yyyy-mm-dd): ")

        return metadata


class PDFPage:
    """Represents a page in a scraped PDF document"""

    page_number: int
    title: Title
    elements: List[Element]
    sections: List[PDFSection]
    text_elements: List[Union[NarrativeText, ListItem]]

    def __init__(self, page_number: int, elements: List[Element]) -> None:
        self.page_number = page_number
        self.title = first(elements, lambda element: isinstance(element, Title))
        self.elements = elements
        self.text_elements = [element for element in self.elements if PDF.is_good_text_element(element)]
        self.sections = PDF.elements_to_sections(
            self.elements
        )  # caution as sections may run over pages; so the list of all PDFPage sections may not be the same
        # as the result of elements_to_sections for all document elements

    def __repr__(self) -> str:
        """Self-explanatory"""
        string = f'\n\n\n--------------p{str(self.page_number)}--{self.title.text if self.title else ""}------------\n'
        string += "\n\n" + "\n\n".join(["\t\t" + str(section) for section in self.sections])
        string += "\n\n\t\tTEXT ELEMENTS NOT ASSIGNED TO A SECTION"
        string += "\n\n" + "\n\n".join(
            ["\t\t" + element.text for element in self.text_elements_not_assigned_to_section]
        )
        string += "\n---------------------------------"
        return string

    # rejection reasons
    @property
    def has_minimal_contents(
        self,
    ) -> bool:
        """Test whether a page only has very short sections, not typical of properly formed paragraphs"""
        if self.sections:
            sections_are_really_short = all(len(section.sentences) <= 1 for section in self.sections)
            return sections_are_really_short
        else:
            return False  # if it has no sections that's a different rejection reason

    @property
    def has_no_good_text_elements(self) -> bool:
        """Test whether a page has no elements which are considered desirable text elements indicating proper content"""
        return not bool(self.text_elements)

    @property
    def is_credits_page(self) -> bool:
        """Test whether a page is the credits page"""
        has_authorship = self.title and re.match("CREATED BY", self.title.text)
        has_copyright_notice = any(
            re.search("This work is copyright Nesta licensed", element.text) for element in self.elements
        )
        return has_authorship and has_copyright_notice

    @property
    def is_divider_page(self) -> bool:
        """
        Test whether a page is a 'divider' page, i.e. a page often used to divide the sections of the report.
        Usually just has the report section number and title.
        """  # noqa
        has_section_number = (
            len(
                [
                    element
                    for element in self.elements
                    if type(element) is Text and re.match(r"\d+$", element.text.strip())
                ]
            )
            == 1
        )
        has_section_title = (
            len([element for element in self.elements if isinstance(element, (Title, NarrativeText))]) == 1
        )  # title may be misclassified as NarrativeText, but it should be the only one on the page
        return has_section_number and has_section_title

    @property
    def is_report_table_of_contents(self) -> bool:
        """Test whether a page looks like the table of contents of a standard report, or part of it"""
        foreword_item = first(
            self.elements, lambda element: re.match("Foreword", element.text)
        )  # could be Foreword or Forewords
        exec_summ_item = first(self.elements, lambda element: re.match("Executive [sS]ummary", element.text))
        return foreword_item and exec_summ_item and len(self.enumerated_items) >= 2

    @property
    def is_table_of_contents(self) -> bool:
        """Test whether a page looks like the table of contents, or part of it"""
        if self.is_report_table_of_contents:
            return True
        else:
            has_floating_numbers = any(
                type(element) is Text and re.match(r"(\d+ )+\d+$", element.text.strip()) for element in self.elements
            )  # lists of numbers indicating page numbers can be found on table of content pages in Text elements
            return has_floating_numbers and len(self.enumerated_items) >= 2

    @property
    def is_title_page(self) -> bool:
        """Test whether a page is the title page"""
        return self.page_number == 1 and self.title

    # test rejection reasons
    @property
    def is_bad_page(self) -> bool:
        """Test whether a page is undesirable content according to one of the tests listed above"""

        rejection_reasons = [
            "is_title_page",
            "is_credits_page",
            "is_table_of_contents",
            "is_divider_page",
            "has_minimal_contents",
            "has_no_good_text_elements",
        ]
        for rejection_reason in rejection_reasons:
            if getattr(self, rejection_reason):
                logger.info(f"Page {self.page_number} has been rejected for the following reason: {rejection_reason}")
                return True
        return False

    @property
    def enumerated_items(self) -> List[Element]:
        """Return any enumerated items which could be sections listed on a table of contents page"""
        return [element for element in self.elements if re.match(r"0\d\s+[A-Za-z]+", element.text)]

    @property
    def text_elements_assigned_to_section(self) -> List[Union[NarrativeText, ListItem]]:
        """Return any text-like elements assigned to a section"""
        return sum([section.text_elements for section in self.sections], [])

    @property
    def text_elements_not_assigned_to_section(self) -> List[Union[NarrativeText, ListItem]]:
        """Return any text-like elements not assigned to a section"""
        text_elements_assigned_to_section_ = self.text_elements_assigned_to_section
        return [
            text_element
            for text_element in self.text_elements
            if text_element not in text_elements_assigned_to_section_
        ]


class PDFSection:
    """
    Represents a section in a scraped PDF document
    A section is defined here as a set of text element items following a title
    Lots of elements are regularly misidentified as titles, so this doesn't work that well,
    particularly in PDFs which are not stand report formats, but more like slidedecks
    It does however seem to identify the start of sections like Executive Summary and Endnotes/References in standard report formats reliably,
    which is useful for getting rid of some types of undesirable text
    """  # noqa

    title: Title
    text_elements: List[Union[NarrativeText, ListItem]]
    sentences_: Optional[List[str]] = None

    def __init__(self, title: Title, text_elements: List[Union[NarrativeText, ListItem]]) -> None:
        self.title = title
        self.text_elements = text_elements

    def __eq__(self, other: object) -> bool:
        """Self-explanatory"""
        if isinstance(other, PDFSection):
            return False
        return self.text == other.text and self.title == other.title

    def __repr__(self) -> str:
        """Self-explanatory"""
        string = f"\n\n\n--------------{self.title.text}--------------\n"
        string += "\n\n" + self.text
        string += "\n---------------------------------"
        return string

    @property
    def is_end_section(self) -> bool:
        """Test whether the section is the Endnotes or References section"""
        return bool(re.match("endnotes|references", self.title.text.lower()))

    @property
    def is_executive_summary(self) -> bool:
        """Test whether the section is the Eexecutive Summary"""
        return bool(re.match("executive summary", self.title.text.lower()))

    @property
    def sentences(self) -> List[str]:
        """Split the section into sentences, store these and return the result; or return a previously stored result"""
        if self.sentences_ is None:  # only do it once for efficiency
            self.sentences_ = []
            for text_element in self.text_elements:
                self.sentences_ += sent_tokenize(text_element.text)
        return self.sentences_

    @property
    def text(self) -> str:
        """Return the section text"""
        return " ".join(
            [element.text for element in [self.title] + self.text_elements]
        )  # text elements can often be misidentified as title elements, so safest to include titles


if __name__ == "__main__":

    paths = ["google_api/downloaded.pdf"]

    for path in paths:

        pdf = PDF(path)

        for element in pdf.elements:
            print(type(element), element, "\n\n")  # noqa

    # for testing and development

    if False:

        page_number = 6
        page = pdf.pages[page_number - 1]
        logger.info(page.is_bad_page)

        for element in page.elements:
            logger.info(element.__class__, element.text)

    if False:
        logger.info("Here's what the document elements looked like:")

        for element in pdf.elements:
            info = element.__dict__
            info["metadata"] = element.metadata.__dict__
            logger.info("\n\n", element.__class__.__name__.upper(), info)
