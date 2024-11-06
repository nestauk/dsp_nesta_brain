import asyncio
import logging
import os
import sys

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
from pdf2image.exceptions import PDFInfoNotInstalledError
from retrieval.db.schema import Chunk
from retrieval.db.schema import Document as LanceDocument
from scraping.scrape import html_to_text
from scraping.scrape import search_query_to_scraped_data
from scraping.scrape_pdf import PDF


_prefix = "2024-10-29"
WEBSITE_DATA_PATH = PROJECT_DIR / f"scraping/data/website_{_prefix}"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100

OPENAI_ENCODING = "cl100k_base"

# OpenAI limits
request_count = {}
RPM_RATE_LIMIT = 10000
TPM_RATE_LIMIT = 5e6

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

    if not request_count:
        request_count = {"time": datetime.now(), "N_requests": [], "N_tokens": [], "cum_N_tokens": 0}
        seconds_since_count_start = None
    elif seconds_since_count_start >= 60:
        request_count["time"] = datetime.now()
        request_count["N_requests"] = []
        request_count["N_tokens"] = []
        # do not reset cum_N_tokens
        seconds_since_count_start = None

    request_count["N_requests"] += [len(texts)]
    N_tokens_ = sum([N_tokens(text) for text in texts])
    request_count["N_tokens"] += [N_tokens_]
    request_count["cum_N_tokens"] += N_tokens_

    cumulative_cost_estimate = round(request_count["cum_N_tokens"] * 0.02 / 1e6, 2)
    logger.info(f"Cumulative cost estimate: ${cumulative_cost_estimate}")

    return seconds_since_count_start


async def throttle(texts: List[str]) -> None:
    """If embeddings model rate limits are exceeded, wait until sufficient time has passed"""
    # this won't work well for larger batch sizes, but unfortunately there isn't really time to troubleshoot and improve it
    # I still get API error messages back with batch_size >= 100 but that can't be due to hitting the rate limit
    # future users may want to improve on it

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
            msg = "About to exceed OpenAI embeddings requests per minute rate limit ... sleeping for {sleep_time} seconds"
        elif sum(request_count["N_tokens"]) >= TPM_RATE_LIMIT:
            msg = "About to exceed OpenAI embeddings token per minute rate limit ... sleeping for {sleep_time} seconds"

        if msg:
            sleep_time = 60 - seconds_since_count_start
            logger.info(msg.format(sleep_time=sleep_time))
            await asyncio.sleep(sleep_time)


def doc_already_in_db(doc: LangchainDocument) -> bool:
    """Determine whether chunks from a source document have already been added to the database"""

    results = chunk_table.search().where(f'source.location = "{doc.metadata["location"]}"').to_list()
    return bool(results)


def chunk_already_in_db(chunk: LangchainDocument) -> bool:
    """Determine whether identical chunks have already been added to the database, because PDFs may be duplicated across the site.
    Chunking strategy should have been the same.
    """  # noqa

    results = chunk_table.search().where(f'text == "{chunk.page_content}"').limit(1).to_list()
    return bool(results), results[0].source.location if results else None


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

    if documents and not docs_split:
        raise Exception(f"Investigate why you have zero chunks for {len(documents)} documents")

    logger.info(f"Fetching embeddings for {len(docs_split)} chunks ...")
    tasks = []
    for i, chunk in enumerate(docs_split):

        new_source = i == 0 or (i > 0 and docs_split[i - 1].metadata["location"] != chunk.metadata["location"])
        if new_source:
            source = [source for source in sources if source.location == chunk.metadata["location"]][0]
            order_index = 1

            skip_source = False
            source_is_pdf = source.location[-4:] == ".pdf"
            if source_is_pdf:
                skip_source, existing_location = chunk_already_in_db(chunk)
            if skip_source:
                logger.info(
                    f"Skipping PDF {source.location} as it already seems to be in the DB with location: {existing_location}"
                )

        if not skip_source:
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

    already_in_db = [doc for doc in documents if doc_already_in_db(doc)]
    if already_in_db:
        N_in_db = len(already_in_db)

        if replace:
            logger.info(f"{N_in_db} documents were already in the database and will be replaced")
            for doc in already_in_db:
                chunk_table.delete(f'source.location = "{doc.metadata["location"]}"')
                document_table.delete(f'location = "{doc.metadata["location"]}"')

        else:
            logger.info(
                f"{N_in_db} documents were already in the database: {len(documents) - N_in_db} remain to be added"
            )
            documents = [doc for doc in documents if doc not in already_in_db]

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
        logger.info(f"Ingested {len(lance_documents)} Documents and {len(chunks)} Chunks to the database")
        document_table.add(lance_documents)
        chunk_table.add(chunks)

    else:
        logger.info("No documents or chunks to ingest to the database")

    logging.getLogger("httpx").setLevel(logging.INFO)


# if scraping/ingesting from entire Nesta website data dump
def webpages_to_ingested_data(
    uids: Optional[List[str]] = None, df: Optional[pd.DataFrame] = None, replace: bool = False
) -> None:
    """Convert dumped Nesta webpages into LangchainDocuments and ingest"""

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

        if i % 25 == 0:
            logger.info(f"Scraping webpage in row {i}")

        page_path = WEBSITE_DATA_PATH / (row["uid"] + ".txt")
        if not os.path.exists(page_path):
            raise Exception(f"No scraped webpage for row index {i}, url {row['url']}, uid {row['uid']}")

        with open(page_path, "r") as f:
            html = f.read()
        text = html_to_text(html)

        if text:
            metadata = row["web_metadata"]
            if type(metadata) is list:  # web_metadata is somtimes a list with a single dict element rather than a dict
                metadata = metadata[0]
            metadata.update(
                {field: row[field] for field in ["url", "rank", "views"]}
            )  # also add these fields to what will be the LangchainDocument and LanceDocument metadata

            metadata["location"] = metadata.pop("url")
            doc = LangchainDocument(page_content=text, metadata=metadata)
            docs.append(doc)

        else:
            logger.info(
                f'Webpage {row["url"]} did not seem to have any text'
            )  # this is the case for some types of long read e.g. https://www.nesta.org.uk/feature/mapping-early-years-practice/

    if docs:
        ingest(docs, replace=replace)

    else:
        logger.info("No docs to ingest!")


# if scraping/ingesting PDFs from entire Nesta website data dump
def pdfs_to_ingested_data(df: pd.DataFrame, replace: bool = False) -> None:
    """Convert dumped Nesta website PDFs into LangchainDocuments and ingest"""

    pdf_dir_path = WEBSITE_DATA_PATH / "pdf_files"

    docs = []
    for i, row in df.iterrows():

        if i % 25 == 0:
            logger.info(f"Row index {i}")

        file_name_and_link_tuples = [(row["pdf_files"][i], link) for i, link in enumerate(row["pdf_links"])]
        nesta_file_name_and_link_tuples = [
            (file_name, link) for file_name, link in file_name_and_link_tuples if "https://nesta.org.uk" in link
        ]

        for file_name, link in nesta_file_name_and_link_tuples:
            path = pdf_dir_path / file_name

            logging.info(f"Opening {file_name}")
            os.system(f"open {path}")  # nosec
            os.system(f'open {row["url"]}')  # nosec

            if input(f'Scrape {file_name}? (any key except enter = "yes")') != "":
                try:
                    pdf = PDF(path, linking_url=row["url"])
                    text = pdf.filtered_text
                except PDFInfoNotInstalledError as e:
                    logging.info(f"Following error frmo trying to read PDF: {e} ... skipping")
                    text = None

                if text:

                    if any(doc.page_content == text for doc in docs):
                        logging.info(f"PDF {file_name} has already been ingested this batch")

                    else:

                        web_metadata = row["web_metadata"]
                        if type(web_metadata) is list:
                            web_metadata = web_metadata[0]
                        metadata = pdf.guess_metadata(date_guess=web_metadata.get("publishDate"), indent="\t")
                        metadata["location"] = link
                        doc = LangchainDocument(page_content=text, metadata=metadata)
                        docs.append(doc)

                else:
                    logger.info(f"\nPDF {file_name} did not seem to have any text – ignoring")

            sys.stdout.write("\n")  # there may be quite a lot of info messages generated by PDF operations
            # this gap in the messages helps keep it readable

    if docs:
        ingest(docs, replace=replace)

    else:
        logger.info("No PDF-derived docs to ingest for this batch")


# if scraping/ingesting from search results
def search_query_to_ingested_data(query: str, site_url: str, replace: bool = False, **kwargs) -> bool:
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

    ingest(docs, replace=replace)

    return bool(scraped_data)


if __name__ == "__main__":

    # SETTINGS
    mode = "web_dump"  # if 'web_search', do a web search, scrape and ingest the results
    # if 'web_dump', ingest data which has already been downloaded from the Nesta website
    possible_modes = ["web_dump", "web_search"]
    replace = False  # if True, if the document already exists in the DB, any chunks derived from it will be deleted and replaced

    # settings relevant to web_dump mode
    pdf_mode = True  # scrape PDFs rather than webpages
    metadata_path = WEBSITE_DATA_PATH / "metadata.jsonl"
    start_index = (
        int(sys.argv[1]) if len(sys.argv) > 1 else 0
    )  # the row of metadata.jsonl to start ingesting; everything prior to this will be ignored
    batch_size = 10  # the number of webpages to ingest at a time

    # settings releant to web_search mode
    query = "Centre for Collective Intelligence Design"
    site_url = "nesta.org.uk"
    subdirectories = sorted(
        ["toolkit", "team", "report", "project", "press-release", "jobs", "feature", "event", "blog"]
    )  # optional

    if mode not in possible_modes:
        raise Exception(f"""mode must be one of the following:{', '.join([f"'{mode}'" for mode in possible_modes])}""")

    if mode == "web_dump":
        # if scraping/ingesting from entire Nesta website data dump

        metadata_df = pd.read_json(metadata_path, lines=True)
        n_rows = metadata_df.shape[0]

        for index in list(range(start_index, n_rows, batch_size)):
            df = metadata_df.iloc[index : (index + batch_size)]

            if pdf_mode:
                pdfs_to_ingested_data(df, replace=replace)
            else:
                webpages_to_ingested_data(df=df, replace=replace)

    elif mode == "web_search":
        # if scraping from web

        if subdirectories:
            urls = [site_url + "/" + subdirectory for subdirectory in subdirectories]
        else:
            urls = [site_url]

        for url in urls:

            for start in list(
                range(0, 100, 10)
            ):  # the start parameter specifies which result set to return from Google Programmable Search;
                # 0 = first set of 10 results, 10 = the next set of 10 results, etc.
                logging.info(f"\nGoogle search result set url = {url}, start = {start}")
                results_returned = search_query_to_ingested_data(query, url, start=start, replace=replace)
                if not results_returned:
                    break
