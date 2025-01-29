import asyncio
import logging
import os
import re
import sys

from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union
from typing import get_args

import lancedb
import pandas as pd
import retrieval.db.ingest.ingest as ing

from bs4 import BeautifulSoup
from bs4.element import Tag
from config import DB_PATH
from dsp_nesta_brain import PROJECT_DIR
from dsp_nesta_brain import logger
from langchain.docstore.document import Document as LangchainDocument
from langdetect import detect
from pdf2image.exceptions import PDFInfoNotInstalledError
from retrieval.db.schema.nesta_brain import Chunk
from retrieval.db.schema.nesta_brain import Document as LanceDocument
from retrieval.db.schema.nesta_brain import MissionProject
from scraping.scrape import html_to_text
from scraping.scrape import search_query_to_scraped_data
from scraping.scrape_pdf import PDF
from utils import unique


_prefix = "2024-10-29"
WEBSITE_DATA_PATH = PROJECT_DIR / f"scraping/data/website_{_prefix}"
PDF_PATH = WEBSITE_DATA_PATH / "pdf_files"
NESTA_SITE_URL = "https://nesta.org.uk"


db = lancedb.connect(DB_PATH)
document_table = db.open_table("document")
chunk_table = db.open_table("chunk")


def doc_already_in_db(doc_or_location: Union[LangchainDocument, str]) -> bool:
    """Determine whether chunks from a source document have already been added to the database"""

    if isinstance(doc_or_location, LangchainDocument):
        location = doc_or_location.metadata["location"]
    elif isinstance(doc_or_location, str):
        location = doc_or_location

    results = chunk_table.search().where(f'source.location = "{location}"').to_list()
    return bool(results)


def chunk_already_in_db(*args) -> bool:
    """Determine whether identical chunks have already been added to the database.
    Chunking strategy should have been the same.
    """  # noqa

    results = ing.chunk_already_in_db(*args)
    return bool(results), results[0].source.location if results else None


async def chunk_to_Chunk(chunk: LangchainDocument, order_index: int, source: LanceDocument) -> Chunk:
    """
    Convert a Langchain chunk (as returned from a text splitter) into an object
    of the Chunk class which can be ingested into the DB
    (including deriving an embedding for the Chunk)
    """  # noqa
    return await ing.chunk_to_Chunk(chunk, order_index=order_index, source=source)


async def documents_to_Chunks(documents: List[LangchainDocument], sources: List[LanceDocument]) -> List[Chunk]:
    """
    Split Langchain documents into chunks and convert these into objects
    of the Chunk class which can be ingested into the DB
    """  # noqa

    docs_split = ing.split_documents(documents)

    tasks = []
    for i, chunk in enumerate(docs_split):

        new_source = i == 0 or (i > 0 and docs_split[i - 1].metadata["location"] != chunk.metadata["location"])
        if new_source:
            source = [source for source in sources if source.location == chunk.metadata["location"]][0]
            order_index = 1
            existing_chunk_count = {}

            skip_source = False
            source_is_pdf = source.location[-4:] == ".pdf"
            if source_is_pdf:
                _, existing_location = chunk_already_in_db(chunk)
                existing_chunk_count[existing_location] = (existing_chunk_count.get(existing_location) or 0) + 1
                is_duplicate = (
                    existing_chunk_count[existing_location] >= 2
                )  # there may be the occasional paragraph which is in
                # more than one document, so make the rule there needs to be two chunks
                # before the document is considered a duplicate
                skip_source = is_duplicate
            if skip_source:
                logger.info(
                    f"Skipping PDF {source.location} as it already seems to be in the DB with location: {existing_location}"
                )

        if not skip_source:
            task = asyncio.create_task(chunk_to_Chunk(chunk, order_index, source))
            tasks.append(task)
            order_index += 1

    await ing.throttle(request_counter, [chunk.page_content for chunk in docs_split])
    logger.info(f"Fetching embeddings for {len(docs_split)} chunks ...")
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
        if chunks:
            logger.info(f"Ingested {len(lance_documents)} Document(s) and {len(chunks)} Chunks into the database")
            document_table.add(lance_documents)
            chunk_table.add(chunks)
        else:
            logger.info(f"No chunks from document(s) {lance_documents} into ingest to the database")

    else:
        logger.info("No documents or chunks to ingest to the database")

    logging.getLogger("httpx").setLevel(logging.INFO)


# if scraping/ingesting from entire Nesta website data dump
def webpages_to_ingested_data(
    uids: Optional[List[str]] = None,
    df: Optional[pd.DataFrame] = None,
    replace: bool = False,
) -> None:
    """Convert dumped Nesta webpages into LangchainDocuments and ingest"""

    if not uids and df is None or (uids and df is not None):
        raise Exception(
            "You must provide EITHER a list of website UIDs or a pandas DataFrame to webpages_to_ingested_data"
        )

    if uids:
        METADATA_PATH = WEBSITE_DATA_PATH / "metadata.jsonl"  # noqa
        metadata_df = pd.read_json(METADATA_PATH, lines=True)
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


def find_download_button_links(row: pd.Series, soup: BeautifulSoup) -> Union[None, str]:
    """Find one or more red buttons indicating a main downloadable PDF on the page and extract the link"""

    def get_url_title(link: Tag) -> Tuple[str]:
        url = link["href"]
        url = url.replace("https://www.nesta.org.uk", NESTA_SITE_URL)
        if NESTA_SITE_URL not in url:
            url = NESTA_SITE_URL + url  # no need for '/'
        document_title = None
        if link.get("onclick"):
            document_title_match = re.search("'documentTitle': '([^']+)',", link["onclick"])
            if document_title_match:
                is_bad_title = re.search(r"\.(pdf|docx?)$", document_title_match.group(1))
                if not is_bad_title:
                    document_title = document_title_match.group(1)
        return url, document_title

    def is_welsh(link: Tag, title: Optional[str] = None) -> bool:
        return detect(link.getText().strip()) == "cy" or (title and (detect(title) == "cy"))

    def is_bad_link(link: Tag) -> bool:
        return link.getText() in ["Register for the event"]

    download_button_divs = soup.find_all("div", {"class": "page-heading__download-item"}) or soup.find_all(
        "div", {"class": "document-cta__item"}
    )

    if download_button_divs:
        download_button_links = [
            ele for ele in [div.find("a", {"class": "btn--primary"}) for div in download_button_divs] if ele
        ]
        if download_button_links:
            links = [(link, get_url_title(link)) for link in download_button_links]
            return [
                (url, title)
                for link, (url, title) in links
                if not is_bad_link(link) and not is_welsh(link, title=title)
            ]

    logger.info(
        f'Was not able to identify the main download button link(s) for webpage {row["url"]} from pdf_links: {row["pdf_links"]}'
    )

    return []


def is_good_link(link: str) -> bool:
    """Test whether the link to a PDF is what we want: current sole criterion is that it is on the Nesta website"""  # noqa

    return NESTA_SITE_URL in link


# if scraping/ingesting PDFs from entire Nesta website data dump
def pdfs_to_ingested_data(
    df: pd.DataFrame, replace: bool = False, download_button_pdf_only: bool = False, cautious: bool = False
) -> None:
    """Convert dumped Nesta website PDFs into LangchainDocuments and ingest"""

    docs = []
    for i, row in df.iterrows():

        logger.info(f"Row index {i}")

        file_names = {link: row["uid"] + "_" + re.split("/", link)[-1] for link in row["pdf_links"]}

        #  print("\n\n", file_names, "\n\n")

        webpage_path = WEBSITE_DATA_PATH / (row["uid"] + ".txt")
        with open(webpage_path, "r") as f:
            html = f.read()
        _, soup = html_to_text(html, return_soup=True)

        if download_button_pdf_only:
            button_links_doc_titles = find_download_button_links(row, soup)

            #   print("\n\n", button_links_doc_titles, "\n\n")

            desirable_file_name_and_link_tuples = [  # stores the file names and links just of the PDFs we're interested in
                # according to some criterion – here the criterion is that the link is
                # contained in a download button (indicating a major publication)
                (file_names.get(link), link, title_guess)
                for link, title_guess in button_links_doc_titles
            ]

        else:
            desirable_file_name_and_link_tuples = [  # see comment above. Here the criterion is simply that the PDF
                # is on the Nesta website and not an external website
                (
                    file_name,
                    link,
                    row["web_metadata"]["title"],
                )  # web metadata title is used as one of the guesses of the title of the PDF
                for link, file_name in file_names.items()
                if is_good_link(link)
            ]

        for file_name, link, title_guess in desirable_file_name_and_link_tuples:

            if not doc_already_in_db(link):
                if file_name:
                    path = PDF_PATH / file_name
                else:
                    path = link

                if cautious:  # if being cautious, you will be asked to decide whether you want to scrape the PDF and
                    # whether the metadata guesses are correct. This opens the PDF and its corresponding
                    # webpage for examination
                    logging.info(f"Opening {file_name}")
                    os.system(f"open {path}")  # nosec
                    os.system(f'open {row["url"]}')  # nosec

                if not cautious or input(f'Scrape {file_name or path}? (any key except enter = "yes")') != "":
                    try:
                        pdf = PDF(path, linking_url=row["url"])
                        text = pdf.filtered_text
                    except PDFInfoNotInstalledError as e:
                        logging.info(f"Following error from trying to read PDF: {e} ... skipping")
                        text = None

                    if text:

                        if any(doc.page_content == text for doc in docs):
                            logging.info(f"PDF {file_name} has already been ingested this batch")

                        else:

                            web_metadata = row["web_metadata"]
                            if type(web_metadata) is list:
                                web_metadata = web_metadata[0]
                            title_guesses = unique([soup.find("title").getText().replace(" | Nesta", ""), title_guess])
                            metadata = pdf.guess_metadata(
                                title_guess=title_guesses,
                                date_guess=web_metadata.get("publishDate"),
                                cautious=cautious,
                                indent="\t",
                            )
                            metadata["location"] = link
                            doc = LangchainDocument(page_content=text, metadata=metadata)
                            docs.append(doc)

                    else:
                        logger.info(f"\nPDF {file_name} did not seem to have any text – ignoring")
            else:
                logger.info(f"PDF {link} was already in the database")

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
    mode_literal = Literal["web_dump", "web_search", "from_csv"]
    mode: mode_literal = "from_csv"  # if 'web_search', do a web search, scrape and ingest the results
    # if 'web_dump', ingest data which has already been downloaded from the Nesta website
    # if 'from_csv', ingest data from a CSV
    replace = False  # if True, if the document already exists in the DB, any chunks derived
    # from it will be deleted and replaced

    # settings relevant to web_dump mode
    pdf_mode = True  # scrape PDFs rather than webpages
    download_button_pdf_only = True  # only scrape PDfs if they are a major research output indicated on the page
    # by being downloadable by clicking a big red button
    cautious = False  # ask whether you want to scrape the PDF and whether the metadata guesses are correct
    METADATA_PATH = WEBSITE_DATA_PATH / "metadata.jsonl"
    start_index = (
        int(sys.argv[1]) if len(sys.argv) > 1 else 0
    )  # the row of metadata.jsonl to start ingesting; everything prior to this will be ignored
    batch_size = 1  # the number of webpages to ingest at a time

    # settings relevant to web_search mode
    query = "Centre for Collective Intelligence Design"
    site_url = NESTA_SITE_URL
    subdirectories = sorted(
        ["toolkit", "team", "report", "project", "press-release", "jobs", "feature", "event", "blog"]
    )  # optional

    # settings relevant to from_csv mode
    CSV_PATH = PROJECT_DIR / "data/Mission Project List.csv"
    schema_class = MissionProject

    # global variable
    request_counter = ing.RequestCounter()

    if mode not in get_args(mode_literal):
        raise Exception(
            f"""mode must be one of the following:{', '.join([f"'{mode}'" for mode in get_args(mode_literal)])}"""
        )

    if mode == "web_dump":
        # if scraping/ingesting from entire Nesta website data dump

        metadata_df = pd.read_json(METADATA_PATH, lines=True)
        downloaded = metadata_df["_status_code"].apply(lambda val: val == 200)
        metadata_df = metadata_df[downloaded]
        n_rows = metadata_df.shape[0]

        for index in list(range(start_index, n_rows, batch_size)):
            df = metadata_df.iloc[index : (index + batch_size)]

            if pdf_mode:
                pdfs_to_ingested_data(
                    df, replace=replace, download_button_pdf_only=download_button_pdf_only, cautious=cautious
                )
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

    elif mode == "from_csv":
        # if ingesting data from a csv

        def chunk_already_in_db(chunk: LangchainDocument) -> bool:  # noqa
            """Determine whether identical chunks have already been added to the database.
            Chunking strategy should have been the same.
            """  # noqa

            where_condition = f'code == "{chunk.metadata["code"]}"'
            results = ing.chunk_already_in_db(chunk, where_condition=where_condition)
            return bool(results)

        async def chunk_to_Chunk(chunk: LangchainDocument, ingestion: bool = True) -> Chunk:  # noqa
            """
            Convert a Langchain document into an object
            of the Chunk class which can be ingested into the DB
            (including deriving an embedding for the Chunk)
            """  # noqa

            try:
                return await ing.chunk_to_Chunk(chunk, ingestion=ingestion, **chunk.metadata)
            except Exception as e:
                return e

        ing.csv_rows_to_ingested_data(
            CSV_PATH, 0, None, identifier="code", Chunk_func=chunk_to_Chunk, chunk_presence_test=chunk_already_in_db
        )
