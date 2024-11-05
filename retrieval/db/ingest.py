import asyncio
import logging
import os

from datetime import datetime
from typing import List
from typing import Optional
from typing import Union

import lancedb
import pandas as pd
import tiktoken

from config import DB_PATH
from dotenv import load_dotenv
from dsp_nesta_brain import PROJECT_DIR
from dsp_nesta_brain import logger
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import CharacterTextSplitter
from openai import AsyncOpenAI
from retrieval.db.schema import Chunk
from retrieval.db.schema import Document as LanceDocument
from scraping.scrape import html_to_text
from scraping.scrape import search_query_to_scraped_data


_prefix = "2024-10-29"
WEBSITE_DATA_PATH = PROJECT_DIR / f"scraping/data/website_{_prefix}"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100

OPENAI_ENCODING = "cl100k_base"

# OpenAI limits
request_count = {}
RPM_RATE_LIMIT = 3000
TPM_RATE_LIMIT = 1e6

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

db = lancedb.connect(DB_PATH)
document_table = db.open_table("document")
chunk_table = db.open_table("chunk")


def N_tokens(text: str) -> int:
    """Extimate the number of tokens in text"""
    return len(tiktoken.get_encoding(OPENAI_ENCODING).encode(text))


def update_request_count(texts: List[str]) -> Union[int, None]:
    """Keep a record of how many requests and tokens have been sent to the embeddings model"""
    global request_count
    if request_count:
        seconds_since_count_start = (datetime.now() - (request_count["time"])).seconds

    if not request_count or (request_count and seconds_since_count_start >= 60):  # noqa
        request_count = {"time": datetime.now(), "N_requests": [], "N_tokens": []}
        seconds_since_count_start = None

    request_count["N_requests"] += [len(texts)]
    request_count["N_tokens"] += [sum([N_tokens(text) for text in texts])]

    return seconds_since_count_start


async def throttle(texts: List[str]) -> None:
    """If embeddings model rate limits are exceeded, wait until sufficient time has passed"""

    seconds_since_count_start = update_request_count(texts)

    if request_count["N_requests"][-1] >= RPM_RATE_LIMIT:
        raise Exception(f"You cannot ask for {RPM_RATE_LIMIT} or more requests to the embeddings model in one go")
    elif request_count["N_tokens"][-1] >= TPM_RATE_LIMIT:
        raise Exception(
            f"You cannot ask for {TPM_RATE_LIMIT} or more tokens to be sent to the embeddings model in one go"
        )

    if seconds_since_count_start and seconds_since_count_start < 60:
        msg = None
        if sum(request_count["N_requests"]) >= RPM_RATE_LIMIT:
            msg = "About to exceed OpenAI embeddings requests per minute rate limit ... sleeping for {sleep_time_} seconds"
        elif sum(request_count["N_tokens"]) >= RPM_RATE_LIMIT:
            msg = (
                "About to exceed OpenAI embeddings token per minute rate limit ... sleeping for {sleep_time_} seconds"
            )

        if msg:
            logger.info(msg)
            await asyncio.sleep(60 - seconds_since_count_start)


def already_in_db(location: str) -> bool:
    """Determine whether chunks from a source document have already been added to the database"""
    results = chunk_table.search().where(f'source.location = "{location}"').to_list()
    return bool(results)


async def chunk_to_Chunk(chunk: LangchainDocument, order_index: int, source: LanceDocument) -> Chunk:
    """
    Convert a Langchain chunk (as returned from a text splitter) into an object
    of the Chunk class which can be ingested into the DB
    (including deriving an embedding for the Chunk)
    """  # noqa
    # intentionally not using the neater syntax documented by lanceDB which automatically calculates embeddings vectors
    # using model.VectorField() specified in the schema.
    # This is because I had issues getting the nested schema to work with this method.
    async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    result = await async_client.embeddings.create(model="text-embedding-3-small", input=chunk.page_content)
    vector = result.data[0].embedding
    return Chunk(text=chunk.page_content, source=source, vector=vector, order_index=order_index)


async def documents_to_Chunks(documents: List[LangchainDocument], sources: List[LanceDocument]) -> List[Chunk]:
    """
    Split Langchain documents into chunks and convert these into objects
    of the Chunk class which can be ingested into the DB
    """  # noqa
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs_split = text_splitter.split_documents(documents)

    logger.info(f"Fetching embeddings for {len(docs_split)} chunks ...")
    tasks = []
    for i, chunk in enumerate(docs_split):
        new_source = i == 0 or (i > 0 and docs_split[i - 1].metadata["location"] != chunk.metadata["location"])
        if new_source:
            source = [source for source in sources if source.location == chunk.metadata["location"]][0]
            order_index = 1
        task = asyncio.create_task(chunk_to_Chunk(chunk, order_index, source))
        tasks.append(task)
        order_index += 1

    await throttle([chunk.page_content for chunk in docs_split])
    return await asyncio.gather(*tasks)


def ingest(documents: List[LangchainDocument], replace: bool = False) -> None:
    """
    Find out which documents are not already in the database, convert them into
    Document and Chunk data in accordance with the db schema
    and insert this into the database
    """  # noqa

    logging.getLogger("httpx").setLevel(logging.WARNING)

    already_in_db_ = [doc for doc in documents if already_in_db(doc.metadata["location"])]
    if already_in_db_:
        N_in_db = len(already_in_db_)

        if replace:
            logger.info(f"{N_in_db} documents were already in the database and will be replaced")
            for doc in already_in_db_:
                chunk_table.delete(f'source.location = "{doc.metadata["location"]}"').to_list()
                document_table.delete(f'location = "{doc.metadata["location"]}"').to_list()

        else:
            logger.info(
                f"{N_in_db} documents were already in the database: {len(documents) - N_in_db} remain to be added"
            )
            documents = [doc for doc in documents if doc not in already_in_db_]

    if documents:

        lance_documents = [LanceDocument(ingestion=True, **doc.metadata) for doc in documents]
        chunks = asyncio.run(documents_to_Chunks(documents, lance_documents))

        # ====CAUTION====
        # document_table.add(lance_documents) introduces data redundancy in the database
        # and should be removed for later versions.
        # The source field in the chunk table does not link to a Document record.
        # If the title of a record in the document table is updated,
        # the source.title for the relevant chunk records remains the same
        # This is a recipe for mess!
        # I am keeping this in temporarily for purposes of experimentation
        logger.info(f"Ingesting {len(lance_documents)} Documents and {len(chunks)} Chunks to the database")
        document_table.add(lance_documents)
        chunk_table.add(chunks)

    else:
        logger.info("No documents or chunks to ingest to the database")

    logging.getLogger("httpx").setLevel(logging.INFO)


# if scraping/ingesting from entire Nesta website data dump
def webpages_to_ingested_data(
    uids: Optional[List[str]] = None, df: Optional[pd.DataFrame] = None, replace: bool = False
) -> None:
    """Convert dumped Nesta website data into LangchainDocuments and ingest"""

    if not uids and df is None or (uids and df is not None):
        raise Exception(
            "You must provide EITHER a list of website UIDs or a pandas DataFrame to webpages_to_ingested_data"
        )

    if uids:
        metadata_path = WEBSITE_DATA_PATH / "metadata.jsonl"  # noqa
        metadata_df = pd.read_json(metadata_path, lines=True)
        rows = [metadata_df[metadata_df["uid"] == uid].iloc[0] for uid in uids]
        df = pd.DataFrame(rows)

    docs = []
    for i, row in df.iterrows():

        if i > 0 and i % 50 == 0:
            logger.info(f"Row {i}")

        page_path = WEBSITE_DATA_PATH / (row["uid"] + ".txt")
        with open(page_path, "r") as f:
            html = f.read()
        text = html_to_text(html)

        metadata = row["web_metadata"]
        if type(metadata) is list:  # web_metadata is somtimes a list with a single dict element rather than a dict
            metadata = metadata[0]
        metadata.update(
            {field: row[field] for field in ["url", "rank", "views"]}
        )  # also add these fields to what will be the LangchainDocument and LanceDocument metadata

        metadata["location"] = metadata.pop("url")
        doc = LangchainDocument(page_content=text, metadata=metadata)
        docs.append(doc)

    ingest(docs, replace=replace)


# if scraping/ingesting from search results
def search_query_to_ingested_data(query: str, site_url: str, **kwargs) -> bool:
    """Perform a search, scrape the webpages from the search results, and ingest the data"""

    scraped_data = search_query_to_scraped_data(query, site_url, **kwargs)

    docs = []
    for datum in scraped_data:
        # this used to be in a separate function - no longer required
        text = datum.pop("text")
        metadata = datum  # assume everything else is metadata; Lance Document __init__ will put metadata
        # into the right format for the DB and will only use metadata it needs
        doc = LangchainDocument(page_content=text, metadata=metadata)
        docs.append(doc)

    ingest(docs)

    return bool(scraped_data)


if __name__ == "__main__":

    # if scraping/ingesting from entire Nesta website data dump

    replace = False

    metadata_path = WEBSITE_DATA_PATH / "metadata.jsonl"
    metadata_df = pd.read_json(metadata_path, lines=True)

    df = metadata_df.iloc[100:110]

    webpages_to_ingested_data(df=df, replace=replace)

    if False:
        # if scraping from web

        query = "Centre for Collective Intelligence Design"
        site_url = "nesta.org.uk"
        subdirectories = sorted(
            ["toolkit", "team", "report", "project", "press-release", "jobs", "feature", "event", "blog"]
        )

        if query and site_url:

            for subdirectory in subdirectories:

                url = site_url + "/" + subdirectory
                for start in list(
                    range(0, 100, 10)
                ):  # the start parameter specifies which result set to return from Google Programmable Search;
                    # 0 = first set of 10 results, 10 = the next set of 10 results, etc.
                    logging.info(f"\nGoogle search result set url = {url}, start = {start}")
                    results_returned = search_query_to_ingested_data(query, url, start=start, save=True)
                    if not results_returned:
                        break
