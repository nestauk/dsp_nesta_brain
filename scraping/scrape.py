from __future__ import annotations

import datetime as dt
import json
import os
import re

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Dict
from typing import List

import requests

from bs4 import BeautifulSoup
from dsp_nesta_brain import PROJECT_DIR
from dsp_nesta_brain import logger
from scraping.google_search import google_api_call
from utils import unique


DATA_DIR = PROJECT_DIR / "scraping/data"

if TYPE_CHECKING:
    from bs4.element import Tag


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


def html_to_text(html_text: str, return_soup: bool = False) -> Dict:
    """Get text from an HTML string"""
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        divs = unique(soup.find_all(is_good_div))
        texty_bits = unique(
            sum([div.find_all(is_good_p_or_list, recursive=False) for div in divs], [])
        )  # assumes we want text from <p> elements and lists, but not other elements
        # (for the moment – we might want to include headings later)

        text = "\n\n".join([get_text(texty_bit) for texty_bit in texty_bits])

    except Exception as e:
        logger.critical(f"The following error was encountered while scraping:\n{e}")

    if return_soup:
        return text.strip(), soup
    else:
        return text.strip()


def scrape(url: str) -> str:
    """Scrape an individual webpage"""

    try:
        result = requests.get(url)  # nosec
        text, soup = html_to_text(result.text, return_soup=True)

        # metadata
        title = soup.find("title").getText().replace(" | Nesta", "")
        data_layer = extract_data_layer(soup)
        try:
            date_pub = data_layer.pop("publishDate")
            date_pub = dt.datetime.strptime(date_pub, "%Y-%m-%d")
        except Exception:
            logger.warning(f"Webpage {url} had no publication date")
            date_pub = None

    except Exception as e:
        logger.critical(f"The following error was encountered while scraping {url}:\n{e}")

    result = data_layer
    result["text"] = text.strip()
    result["title"] = title
    result["date_pub"] = date_pub

    return result


def extract_pdf_links(soup: BeautifulSoup, base: str = "https://www.nesta.org.uk") -> list:
    """Find all PDF links in the page and return them as a list"""
    # Extract all PDF links
    pdf_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".pdf"):
            # If it's a relative URL, make it absolute
            if not href.startswith("http"):
                href = f"{base}{href}"
            pdf_links.append(href)
    # Remove duplicates
    return list(set(pdf_links))


def download_pdfs(pdf_links: List[str], download_dir: Path = DATA_DIR, suffix: str = None) -> List[str]:
    """Download all PDFs from the list of links"""
    pdf_filenames = []
    suffix = "" if suffix is None else f"{suffix}_"
    if len(pdf_links) > 0:
        for link in pdf_links:
            try:
                pdf_response = requests.get(link, timeout=20)
                if pdf_response.status_code == 200:
                    # Extract the PDF filename from the URL
                    pdf_filename = f"{suffix}{os.path.basename(link)}"
                    with open(download_dir / f"{pdf_filename}", "wb") as f:
                        f.write(pdf_response.content)
                    logger.info(f"Downloaded: {pdf_filename}")
                    pdf_filenames.append(pdf_filename)
                else:
                    logger.info(f"Failed to download {link}")
            except Exception as e:
                logger.info(f"Error downloading {link}: {e}")
    return pdf_filenames


def extract_data_layer(soup: BeautifulSoup) -> Dict:
    """Extract the dataLayer information from a webpage

    Data layer contains structured information about the page, such as the page title,
    publication date, mission area, and authors
    """
    script_tag = soup.find("script", string=re.compile(r"dataLayer\s*=\s*\["))
    dataLayer_text = re.search(r"dataLayer\s*=\s*(\[\{.*?\}\]);", script_tag.string, re.DOTALL)

    if dataLayer_text:
        dataLayer_json = dataLayer_text.group(1)  # Extract the JSON-like string
        dataLayer_json = dataLayer_json.replace("'", '"')  # Replace single quotes with double quotes for JSON parsing
        dataLayer = json.loads(dataLayer_json)  # Parse to Python dict
        return dataLayer
    else:
        return None


def scrape_multiple_pages(urls: List[str], save: bool = False, **kwargs) -> List[str]:
    """Derive a set of Google programmable search results from the query and scrape them"""

    scraped_data = []

    for url in urls:
        scraped_datum = scrape(url)
        scraped_datum["url"] = url
        scraped_data.append(scraped_datum)

        if save:
            uid = url.replace("https://www.nesta.org.uk/", "").replace("/", "-")
            uid = re.sub("-$", "", uid)
            scraped_datum["date_pub"] = (
                datetime.strftime(scraped_datum["date_pub"], "%Y-%m-%d") if scraped_datum["date_pub"] else None
            )
            with open(f"scraping/data/nesta_{uid}.json", "w") as f:
                json.dump(scraped_datum, f)

    return scraped_data


def search_query_to_scraped_data(query: str, site_url: str, save: bool = False, **kwargs) -> List[str]:
    """Derive a set of Google programmable search results from the query and scrape them"""

    scraped_data = []

    search_results = google_api_call(query, site_url, **kwargs) or []

    urls = [search_result.get("link") for search_result in search_results]

    scraped_data = scrape_multiple_pages(urls, save=save)

    return scraped_data


if __name__ == "__main__":

    query = "Centre for Collective Intelligence Design"
    site_url = "nesta.org.uk"
    subdirectories = sorted(
        ["toolkit", "team", "report", "project", "press-release", "jobs", "feature", "event", "blog"]
    )
    webpage_url = "https://www.nesta.org.uk/jobs/product-designer-centre-for-collective-intelligence-design-ccid/"

    if query and site_url:
        # convert a set of Google programmable search results into text
        # just visually inspecting the results for now - the next step will be to vectorize them

        for subdirectory in subdirectories:

            url = site_url + "/" + subdirectory
            for start in list(
                range(0, 100, 10)
            ):  # the start parameter specifies which result set to return from Google Programmable Search;
                # 0 = first set of 10 results, 10 = the next set of 10 results, etc.
                logger.info(f"\nGoogle search result set url = {url}, start = {start}")

            results_returned = search_query_to_scraped_data(query, site_url, save=False)
            if not results_returned:
                break

    elif webpage_url:
        # scrape a single webpage

        text = scrape(webpage_url).get("text")

        logger.info(text)
