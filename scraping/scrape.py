from __future__ import annotations

import datetime as dt
import json
import re

from datetime import datetime
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
        "listing-item__text",
    }
    bad_text = [
        "Join our mailing list to receive the Nesta edit",
        "To contact the Collective Intelligence",
        "Subscribe to our bi-monthly newsletter|Photo credit:",
    ]
    bad_text_regex = "|".join(bad_text)
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

    try:
        result = requests.get(url)  # nosec
        soup = BeautifulSoup(result.text, "html.parser")

        divs = unique(soup.find_all(is_good_div))
        texty_bits = unique(
            sum([div.find_all(is_good_p_or_list, recursive=False) for div in divs], [])
        )  # assumes we want text from <p> elements and lists, but not other elements
        # (for the moment – we might want to include headings later)

        text = "\n\n".join([get_text(texty_bit) for texty_bit in texty_bits])

        # metadata
        title = soup.find("title").getText().replace(" | Nesta", "")
        script = soup.find_all("script")[0]  # publication date should be in the first script element of the page
        try:
            date_pub = re.search("'publishDate': '(\d{4}-\d{2}-\d{2})'", script.getText()).groups()[0]  # noqa
            date_pub = dt.datetime.strptime(date_pub, "%Y-%m-%d")
        except Exception:
            logger.warning(f"Webpage {url} had no publication date")
            date_pub = None

    except Exception as e:
        logger.critical(f"The following error was encountered while scraping {url}:\n{e}")

    return {"text": text.strip(), "title": title, "date_pub": date_pub}


def search_query_to_scraped_data(query: str, site_url: str, save: bool = False, **kwargs) -> List[str]:
    """Derive a set of Google programmable search results from the query and scrape them"""

    scraped_data = []

    search_results = google_api_call(query, site_url, **kwargs) or []

    for search_result in search_results:
        scraped_datum = scrape(search_result)
        scraped_datum["url"] = search_result.get("link")
        scraped_data.append(scraped_datum)

        if save:
            uid = search_result.get("link").replace("https://www.nesta.org.uk/", "").replace("/", "-")
            uid = re.sub("-$", "", uid)
            scraped_datum["date_pub"] = (
                datetime.strftime(scraped_datum["date_pub"], "%Y-%m-%d") if scraped_datum["date_pub"] else None
            )
            with open(f"scraping/data/nesta_{uid}.json", "w") as f:
                json.dump(scraped_datum, f)

    return scraped_data


if __name__ == "__main__":

    query = None  # "Centre for Collective Intelligence Design"
    site_url = None  # "nesta.org.uk"
    webpage_url = "https://www.nesta.org.uk/jobs/product-designer-centre-for-collective-intelligence-design-ccid/"

    if query and site_url:
        # convert a set of Google programmable search results into text
        # just visually inspecting the results for now - the next step will be to vectorize them

        texts = search_query_to_scraped_data(query, site_url)

        for i, text in enumerate(texts):

            file_path = f"scraping/data/test_{i}.txt"
            f = open(file_path, "w")
            f.write(text["text"])
            f.close()

    elif webpage_url:
        # scrape a single webpage

        text = scrape(webpage_url).get("text")

        logger.info(text)
