import asyncio
import datetime
import json
import re

import aiofiles
import aiohttp
import pandas as pd

from bs4 import BeautifulSoup
from dsp_nesta_brain import PROJECT_DIR
from dsp_nesta_brain import logger
from scraping import scrape


BASE_URL = "https://nesta.org.uk"

SITEMAP_PATH = PROJECT_DIR / "data/Nesta_sitemap_2024-10-29.csv"

_prefix = datetime.datetime.now().strftime("%Y-%m-%d")
OUTPUTS_PATH = PROJECT_DIR / f"data/outputs_{_prefix}"
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUTPUTS_PATH / "pdf_files"
PDF_PATH.mkdir(parents=True, exist_ok=True)

METADATA_FILE = OUTPUTS_PATH / "metadata.jsonl"

# Site groups we'll include in scraping
SITE_GROUPS = [
    "blog",
    "report",
    "feature",
    "team",
    "project",
    "event",
    "press-release",
    "project_updates",
    "toolkit",
    "data-visualisation-and-interactive",
    "areas_of_work",
    "case-study",
    "about-us",
    "introduction-our-strategy",
    "about",
    "fairer-start",
    "healthy-life",
    "sustainable-future",
]


def split_url(url: str) -> list:
    """Split a URL into its components"""
    try:
        return [u for u in url.split("/") if len(u) > 0]
    except BaseException:
        return None


def url_to_uid(url: str) -> str:
    """Convert a URL to a UID by removing the base URL and replacing slashes with dashes"""
    uid = url.replace(BASE_URL, "").replace("/", "-")
    uid = re.sub("-$", "", uid)
    return uid


def clean_uid(uid: str) -> str:
    """Remove dash in the beginning of the UID"""
    return re.sub("^-", "", uid)


def load_sitemap(sitemap_path: str) -> pd.DataFrame:
    """Load the Nesta sitemap"""
    return (
        pd.read_csv(SITEMAP_PATH)
        # Get the URL components
        .assign(site_group=lambda df: df.website.apply(split_url))
        .dropna(subset=["site_group"])
        # Remove home page from the list
        .assign(site_group_len=lambda df: df.site_group.apply(len))
        .query("site_group_len > 0")
        .drop(columns=["site_group_len"])
        # Get the site group
        .assign(site_group=lambda df: df.site_group.apply(lambda x: x[0]))
        .drop_duplicates(subset=["website"])
        # Turn views into integers
        .fillna({"views": "0", "active_users": "0"})
        .assign(views=lambda df: df.views.apply(lambda x: x.replace(",", "")).astype(int))
        .assign(uid=lambda df: df.website.apply(url_to_uid))
        .assign(uid=lambda df: df.uid.apply(clean_uid))
        # Remove duplicated UIDs
        .sort_values("rank", ascending=True)
        .drop_duplicates(subset=["uid"], keep="first")
        # Filter by site group
        .query("site_group in @SITE_GROUPS")
    )


def load_existing_metadata() -> set:
    """Load existing UIDs from the metadata file if it exists."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            return {json.loads(line).get("uid") for line in f}
    return set()


async def fetch_data(session: aiohttp.ClientSession, url: str, row: pd.Series) -> None:
    """Fetch website data asynchronously and prepare for saving"""
    try:
        async with session.get(url, timeout=120, headers={"User-Agent": "karlis.kanders@nesta.org.uk"}) as response:
            metadata = {"_status_code": response.status}
            if response.status == 200:
                text = await response.text()

                # Process and prepare metadata
                soup = BeautifulSoup(text, "html.parser")
                pdf_links = scrape.extract_pdf_links(soup, BASE_URL)
                metadata.update(
                    {
                        "url": url,
                        "website": row.website,
                        "uid": row.uid,
                        "site_group": row.site_group,
                        "rank": row["rank"],
                        "views": row.views,
                        "status_code": response.status,
                        "pdf_links": pdf_links,
                        "pdf_files": scrape.download_pdfs(pdf_links, PDF_PATH, suffix=row.uid),
                        "web_metadata": scrape.extract_data_layer(soup),
                    }
                )

                # Save content and metadata
                await save_content_and_metadata(text, metadata, row.uid)
    except asyncio.TimeoutError:
        logger.info(f"Timeout error for {url}")
    except BaseException as e:
        logger.info(f"Error for {url}: {e}")


async def save_content_and_metadata(text: str, metadata: dict, uid: str) -> None:
    """Save both HTML content and metadata asynchronously"""
    # Save HTML text content
    txt_file_path = OUTPUTS_PATH / f"{uid}.txt"
    async with aiofiles.open(txt_file_path, "w") as txt_file:
        await txt_file.write(text)

    # Save metadata to the JSONL file
    async with aiofiles.open(METADATA_FILE, "a") as f:
        await f.write(json.dumps(metadata) + "\n")


async def process_batch(session: aiohttp.ClientSession, batch: pd.DataFrame) -> None:
    """Process a batch of URLs asynchronously"""
    tasks = [fetch_data(session, f"{BASE_URL}{row.website}", row) for _, row in batch.iterrows()]
    await asyncio.gather(*tasks)


async def main() -> None:
    """Scrape Nesta website"""
    # Load the sitemap
    sitemap_df = load_sitemap(SITEMAP_PATH)
    sitemap_df.to_csv(OUTPUTS_PATH / "_input_websites.csv", index=False)

    existing_uids = load_existing_metadata()
    sitemap_df = sitemap_df[~sitemap_df["uid"].isin(existing_uids)]

    # Scrape the website
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(sitemap_df), 10):
            batch = sitemap_df.iloc[i : i + 10]
            logger.info(f"Batch {i // 10 + 1} out of {len(sitemap_df) // 10 + 1}")
            await process_batch(session, batch)
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
