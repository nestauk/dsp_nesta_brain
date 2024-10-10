from __future__ import annotations

import re

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import requests

from bs4 import BeautifulSoup
from dsp_nesta_brain import logger
from scraping.google_search import google_api_call


if TYPE_CHECKING:
    from bs4.element import Tag


def unique(seq: Union[List, Tuple]) -> List:
    """Find unique elements of a sequence and retain order"""
    seen = {}
    result = []
    for item in seq:
        if item in seen:
            continue
        seen[item] = 1
        result.append(item)
    return result


# --- functions for filtering out particular elements that we don't want based on their class, or possibly the text they contain


def is_good_div(tag: Tag) -> bool:
    """Test whether tag is a suitable div"""
    bad_classes = {"newsletter__container", "newsletter__content", "newsletter__small-print-container"}
    return tag.name == "div" and not set(tag.get("class") or {}).intersection(bad_classes)


def is_good_p(tag: Tag) -> bool:
    """Test whether tag is a suitable <p> element"""
    bad_classes = {
        "priority-area__title",
        "share-item__heading",
        "fixed-message__message",
        "print__caption",
        "author-item__job-title",
        "author-item__bio",
        "image-cta__action",
    }
    # 'newsletter__body','newsletter__note'}
    bad_text_regex = "Join our mailing list to receive the Nesta edit"
    is_p = tag.name == "p"
    does_not_have_bad_class = not set(tag.get("class") or {}).intersection(bad_classes)
    does_not_have_bad_text = not re.search(bad_text_regex, tag.getText())
    return is_p and does_not_have_bad_class and does_not_have_bad_text


def is_good_list(tag: Tag) -> bool:
    """Test whether tag is a suitable list"""
    return tag.name in ["ol", "ul"] and all(is_good_li(list_element) for list_element in tag.find_all("li"))


def is_good_li(tag: Tag) -> bool:
    """Test whether tag is a suitable list element"""
    bad_classes = {"app-search__item"}
    return tag.name == "li" and not set(tag.get("class") or {}).intersection(bad_classes)


def is_good_p_or_list(tag: Tag) -> bool:
    """Test whether tag is a suitable <p> element or list"""
    return is_good_p(tag) or is_good_list(tag)


# ---


def get_text(tag: Tag) -> str:
    """Return text from a Tag object; not really needed for <p> elements, but joins list elements together"""

    if tag.name == "p":
        return tag.getText()
    elif tag.name in ["ul", "ol"]:
        return "\n".join([list_element.getText() for list_element in tag.find_all("li")])
    else:
        logger.warning("Unrecognised tag name in get_text")


def scrape(google_search_result_or_url: Union[Dict, str]) -> str:
    """Scrape an individual webpage"""

    if type(google_search_result_or_url) is dict:
        # interpret google_search_result_or_url as a search result
        url = google_search_result_or_url.get("link")
    else:
        # interpret google_search_result_or_url as a url
        url = google_search_result_or_url

    result = requests.get(url)  # nosec
    soup = BeautifulSoup(result.text, "html.parser")

    divs = unique(soup.find_all(is_good_div))
    texty_bits = unique(
        sum([div.find_all(is_good_p_or_list, recursive=False) for div in divs], [])
    )  # assumes we want text from <p> elements and lists, but not other elements
    # (for the moment – we might want to include headings later)

    text = "\n\n".join([get_text(texty_bit) for texty_bit in texty_bits])

    return text.strip()


def search_query_to_texts(query: str, site_url: str, **kwargs) -> List[str]:
    """Derive a set of Google programmable search results from the query and scrape them"""

    texts = []

    search_results = google_api_call(query, site_url, **kwargs)
    for search_result in search_results:
        text = scrape(search_result)
        texts.append(text)

    return texts


if __name__ == "__main__":

    query = None  # "Centre for Collective Intelligence Design"
    site_url = None  # "nesta.org.uk"
    webpage_url = "https://www.nesta.org.uk/jobs/product-designer-centre-for-collective-intelligence-design-ccid/"

    if query and site_url:
        # convert a set of Google programmable search results into text
        # just visually inspecting the results for now - the next step will be to vectorize them

        texts = search_query_to_texts(query, site_url)

        for i, text in enumerate(texts):

            file_path = f"data/test_{i}.txt"
            f = open(file_path, "w")
            f.write(text)
            f.close()

    elif webpage_url:
        # scrape a single webpage

        text = scrape(webpage_url)

        logger.info(text)
