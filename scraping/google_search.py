import os

from typing import Dict
from typing import List

from dotenv import load_dotenv
from dsp_nesta_brain import logger
from googleapiclient.discovery import build


def google_api_call(query: str, site_url: str, start: int = 0, **kwargs) -> List[Dict]:
    """Send a query to Google Programmable Search API and return a list of results

    start: a parameter specifying which result set to return from Google Programmable Search;
           0 = first set of 10 results, 10 = the next set of 10 results, etc.
    """

    load_dotenv()

    full_query = f"{query} site:{site_url}"

    service = build(
        "customsearch", "v1", developerKey=os.getenv("GOOGLE_SEARCH_API_KEY")
    )  # only need the API key if using paid search
    raw_results = service.cse().list(q=full_query, cx=os.getenv("GOOGLE_SEARCH_ID"), start=start, **kwargs).execute()

    search_results = raw_results.get("items")

    if not search_results:
        logger.info(f'No search results for query "{full_query}", start: {start}')

    return search_results


if __name__ == "__main__":

    query = "Centre for Collective Intelligence Design"
    site_url = "nesta.org.uk"

    search_results = google_api_call(query, site_url)

    if not search_results:
        logger.info(f"No search results using query {query}")

    for i, result in enumerate(search_results):
        logger.info(f"\n\n{(i+1)}: {result}")
