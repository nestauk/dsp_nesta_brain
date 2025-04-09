import argparse
import asyncio
import logging
import os
import re
import sys

from datetime import datetime
from enum import Enum
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import lancedb
import pandas as pd
import retrieval.db.ingest.const as const
import retrieval.db.ingest.ingest as ing

from bs4 import BeautifulSoup
from bs4.element import Tag
from config import DB_PATH
from dsp_nesta_brain import PROJECT_DIR
from dsp_nesta_brain import logger
from google_api.drive import READ_ONLY_SCOPES
from google_api.drive import download_pdf
from google_api.drive import drive_service
from google_api.drive import get_file
from google_api.drive import list_files
from langchain.docstore.document import Document as LangchainDocument
from langdetect import detect
from pdf2image.exceptions import PDFInfoNotInstalledError
from retrieval.db.schema.nesta_brain import Chunk as NestaBrainChunk
from retrieval.db.schema.nesta_brain import Document as LanceDocument

# from retrieval.db.schema.nesta_brain import MissionProject
from scraping.pdf.openparse_ import OpenParsePDF
from scraping.pdf.unstructured_ import PDF
from scraping.scrape import html_to_text
from scraping.scrape import scrape
from scraping.scrape import search_query_to_scraped_data
from utils import unique


pause = input


mode_type: Type = Literal["web_dump", "web_search", "given_urls", "from_csv", "from_drive"]  # possible modes

_prefix = "2024-10-29"
WEBSITE_DATA_PATH = PROJECT_DIR / f"scraping/data/website_{_prefix}"
METADATA_PATH = WEBSITE_DATA_PATH / "metadata.jsonl"
PDF_PATH = WEBSITE_DATA_PATH / "pdf_files"
CSV_PATH = PROJECT_DIR / "data/Mission Project List.csv"
NESTA_SITE_URL = "https://nesta.org.uk"
DEFAULT_BATCH_SIZE = 10

DB = lancedb.connect(DB_PATH)


def doc_already_in_db(doc_or_location: Union[LangchainDocument, str]) -> bool:
    """Determine whether chunks from a source document have already been added to the database"""

    if isinstance(doc_or_location, LangchainDocument):
        location = doc_or_location.metadata["location"]
    elif isinstance(doc_or_location, str):
        location = doc_or_location

    results = chunk_table.search().where(f'source.location = "{location}"').to_list()
    return bool(results)


def chunk_already_in_db(chunk: LangchainDocument, **kwargs) -> bool:
    """Determine whether identical chunks have already been added to the database.
    Chunking strategy should have been the same.
    """  # noqa

    if const.CHUNK_TABLE_NAME == "mission_project":

        where_condition = f'''name == "{chunk.metadata.get("Project Name (Asana)") or chunk.metadata.get("name")}"'''
        results = ing.chunk_already_in_db(chunk, where_condition=where_condition, **kwargs)
        return bool(results)

    else:

        results = ing.chunk_already_in_db(chunk)
        return bool(results), results[0].source.location if results else None


async def chunk_to_Chunk(chunk: LangchainDocument, **kwargs) -> const.Chunk:
    """
    Convert a Langchain chunk (as returned from a text splitter) into an object
    of the Chunk class which can be ingested into the DB
    (including deriving an embedding for the Chunk)
    """  # noqa

    if const.CHUNK_TABLE_NAME == "mission_project":

        try:
            return await ing.chunk_to_Chunk(chunk, ingestion=True, **chunk.metadata)
        except Exception as e:
            raise e

    else:

        return await ing.chunk_to_Chunk(chunk, ingestion=True, **kwargs)


async def documents_to_Chunks(documents: List[LangchainDocument], split_documents: bool = True) -> List[const.Chunk]:

    """
    Split Langchain documents into chunks and convert these into objects
    of the Chunk class which can be ingested into the DB
    """  # noqa

    logging.getLogger("openai").setLevel(logging.WARNING)

    sources = [LanceDocument(ingestion=True, **doc.metadata) for doc in documents]

    if split_documents:
        docs_split = ing.split_documents(documents)

        tasks = []
        source_tasks = []
        for i, chunk in enumerate(docs_split):

            # determine whether this chunk is from a different source document to the previous chunk
            # if so, restart the order index
            new_source = i == 0 or (i > 0 and docs_split[i - 1].metadata["location"] != chunk.metadata["location"])
            if new_source:

                # append the tasks for the previous source to the list of tasks
                tasks += source_tasks

                source = [source for source in sources if source.location == chunk.metadata["location"]][0]
                order_index = 1
                existing_chunk_count = {}
                skip_source = False
                # if the document is a PDF, check whether it is already in the DB and skip if so
                # this is because links to the same PDF can be duplicated across webpages
                source_is_pdf = source.location[-4:] == ".pdf"
                source_tasks = []

            if source_is_pdf and not skip_source:
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
                source_tasks = []

            else:
                task = asyncio.create_task(chunk_to_Chunk(chunk, order_index=order_index, source=source))
                source_tasks.append(task)
                order_index += 1

        tasks += source_tasks  # add the final set of source_tasks

        logger.info(f"Fetching embeddings for {len(tasks)} chunks ...")
        chunks = await asyncio.gather(*tasks)

    else:

        chunk_to_Chunk_ = lambda doc: chunk_to_Chunk(  # noqa
            doc, source=LanceDocument(ingestion=True, **doc.metadata)
        )  # order_index is not needed if not splitting
        chunks = await ing.documents_to_Chunks_no_split(
            documents,
            skip_message_format="Skipping document {location} as it already seems to be in the DB",
            chunk_to_Chunk=chunk_to_Chunk_,
        )

    logging.getLogger("openai").setLevel(logging.INFO)

    return chunks, sources


def ingest(documents: List[LangchainDocument], replace: bool = False, **kwargs) -> None:
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

        else:
            logger.info(
                f"{N_in_db} documents were already in the database: {len(documents) - N_in_db} remain to be added"
            )
            documents = [doc for doc in documents if doc not in already_in_db]

    if documents:

        chunks, lance_documents = asyncio.run(documents_to_Chunks(documents, **kwargs))

        if chunks:
            logger.info(f"Ingested {len(chunks)} Chunk(s) from {len(lance_documents)} Document(s) into the database")
            chunk_table.add(chunks)
        else:
            logger.info(f"No chunks from {len(lance_documents)} Document(s) to ingest into the database")

    else:
        logger.info("No documents or chunks to ingest to the database")

    logging.getLogger("httpx").setLevel(logging.INFO)


# if scraping/ingesting from entire Nesta website data dump
def webpages_to_ingested_data(uids: Optional[List[str]] = None, df: Optional[pd.DataFrame] = None, **kwargs) -> None:
    """Convert dumped Nesta webpages into LangchainDocuments and ingest"""

    if not uids and df is None or (uids and df is not None):
        raise Exception(
            "You must provide EITHER a list of website UIDs or a pandas DataFrame to webpages_to_ingested_data"
        )

    if uids:
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
        ingest(docs, **kwargs)

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
    df: pd.DataFrame, download_button_pdf_only: bool = False, cautious: bool = False, **kwargs
) -> None:
    """Convert dumped Nesta website PDFs into LangchainDocuments and ingest"""

    docs = []
    for i, row in df.iterrows():

        logger.info(f"Row index {i}")

        file_names = {link: row["uid"] + "_" + re.split("/", link)[-1] for link in row["pdf_links"]}

        webpage_path = WEBSITE_DATA_PATH / (row["uid"] + ".txt")
        with open(webpage_path, "r") as f:
            html = f.read()
        _, soup = html_to_text(html, return_soup=True)

        web_metadata = row["web_metadata"]
        if type(web_metadata) is list:
            web_metadata = web_metadata[0]

        if download_button_pdf_only:
            button_links_doc_titles = find_download_button_links(row, soup)

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
                    web_metadata["title"],
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
        ingest(docs, **kwargs)

    else:
        logger.info("No PDFs to ingest for this batch")


# if scraping/ingesting from search results
def scraped_data_to_ingested_data(scraped_data: List[Dict], **kwargs) -> bool:
    """Perform a search, scrape the webpages from the search results, and ingest the data"""

    docs = []
    for datum in scraped_data:
        text = datum.pop("text")
        metadata = datum  # assume everything else is metadata; Lance Document __init__ will put metadata
        # into the right format for the DB and will only use metadata it needs
        doc = LangchainDocument(page_content=text, metadata=metadata)
        docs.append(doc)

    ingest(docs, **kwargs)

    return bool(scraped_data)


def search_query_to_ingested_data(
    query: str, site_url: str, replace: bool = False, split_documents: bool = True, **kwargs
) -> bool:
    """Perform a search, scrape the webpages from the search results, and ingest the data"""

    scraped_data = search_query_to_scraped_data(query, site_url, **kwargs)
    return scraped_data_to_ingested_data(scraped_data, replace=replace, split_documents=split_documents)


# if ingesting from Google Drive
def ingest_from_drive(
    file_ids: Optional[List[str]] = None,
    all: bool = False,
    drive_type: Optional[str] = None,
    pdf_parser: str = "openparse",
    **kwargs,
) -> None:
    """Ingest PDFs from Google Drive into the database"""

    def guess_metadata(file_id: str) -> Tuple[str, Union[None, datetime.date]]:
        file_metadata = get_file(file_id, is_pdf=True, silent=True)  # the only useful metadata here is file name
        policy_date_regex = r" - ((\d.. )?[A-Z][a-z]+ \d{4})(.+)?(\.(docx?|pdf))+"
        # Policy document file names generally have the format: name - Month Year(.docx)?.pdf
        title_guess = re.sub(policy_date_regex, "", file_metadata.get("name")).strip().title()  #
        date_guess = None
        match_ = re.search(policy_date_regex, file_metadata.get("name"))
        if match_:
            date_formats = ["%B %Y", "%dth %B %Y", "%dst %B %Y", "%dnd %B %Y", "%drd %B %Y"]
            while not date_guess and date_formats:
                date_format = date_formats.pop(0)
                try:
                    date_guess = datetime.strptime(match_.group(1), date_format).date()
                except ValueError:
                    pass
        return title_guess, date_guess

    if file_ids is None and not all:
        raise Exception("You must provide either a list of file_ids or set all=True")

    if all:
        files = list_files(mimetype="application/pdf")
        file_ids = [file["id"] for file in files]
        logger.info(f"Found {len(file_ids)} PDFs in Google Drive")

    if file_ids:
        logger.info("Setting up connection to Google Drive API")
        service = drive_service(scopes=READ_ONLY_SCOPES)

    for file_id in file_ids:

        location = f"https://drive.google.com/file/d/{file_id}"
        if doc_already_in_db(
            location
        ):  # do here rather than in ingest to avoid having to guess metadata if the document is already in the DB
            logger.info(f"A document with {location} was already in the database ... skipping")

        else:

            pdf_path = "google_api/downloaded.pdf"
            download_pdf(file_id, path=pdf_path, service=service)
            if pdf_parser == "openparse":
                pdf = OpenParsePDF(pdf_path)
                text = pdf.text
            else:
                pdf = PDF(pdf_path)

            text = (
                pdf.text
            )  # use text rather than filtered_text because the formatting of policy documents is different to main reports
            # where sections are identified via titles; titles in policy documents are often not identified

            title_guess, date_guess = guess_metadata(file_id)
            metadata = pdf.guess_metadata(
                title_guess=title_guess, date_guess=date_guess, cautious=True, force_date=True
            )
            metadata["location"] = location
            metadata["drive_type"] = drive_type
            doc = LangchainDocument(page_content=text, metadata=metadata)
            ingest([doc], **kwargs)


class ModeArgEnum(Enum):
    """Specifies the values that --mode can take via the command line"""

    wd = "wd"
    ws = "ws"
    csv = "csv"
    drv = "drv"


class DriveTypeEnum(Enum):
    """Specifies the values that --drive_type (a variable describing the document type on Drive) can take via the command line"""

    policy = "policy"  # add others as needed


if __name__ == "__main__":

    # ---------NON-COMMAND LINE SETTINGS

    # define the class of the chunks to be ingested
    # this should be a class imported from the relevant schema
    # for example, you may be ingesting project data from a CSV file, with its own table in the DB
    # rather than standard documents
    # also specify the relevant table name in the database
    # these constants may be used by other modules so set them as const.var_name
    # rather than as a global variable just for use in this module

    const.Chunk = NestaBrainChunk
    const.CHUNK_TABLE_NAME = "chunk"
    chunk_table = DB.open_table(const.CHUNK_TABLE_NAME)

    # list of subdirectories if used in web_search mode
    subdirectories: List[str] = sorted(
        ["toolkit", "team", "report", "project", "press-release", "jobs", "feature", "event", "blog"]
    )

    # ---------COMMAND LINE ARGUMENTS

    # MODE DEFINITION
    # possible modes and their command line instructions
    # if 'web_dump': ingest data which has already been downloaded from the Nesta website.
    #                Use '-m wd' in the command line.
    # if 'web_search': do a web search, scrape and ingest the results.
    #                  Use '-m ws' in the command line.
    # if 'from_csv': ingest from the CSV file specified by CSV_PATH.
    #                  Use '-m csv' in the command line.
    # if 'from_drive': ingest from Google Drive.
    #                  Use '-m drv' in the command line.
    # if 'given_urls': scrape webpages from a list of known urls otherwise.
    #                  Omit -m and specify the urls via --urls in the command line.
    mode_args_map = {"wd": "web_dump", "ws": "web_search", "csv": "from_csv", "drv": "from_drive"}

    parser = argparse.ArgumentParser()

    # universal arguments
    parser.add_argument("-m", "--mode", type=ModeArgEnum)
    parser.add_argument(
        "-r", "--replace", action="store_true"
    )  # replace flag. If present, if the document already exists in the DB, any chunks derived
    # from it will be deleted and replaced

    # arguments only relevant in web_dump mode
    parser.add_argument("--pdf", action="store_true")  # PDF flag. If present, scrape PDFs rather than webpages
    parser.add_argument(
        "--all", action="store_true"
    )  # all PDFs flag. If present, attempt to scrape all PDFs linked to by a webpage.
    # by default, only scrape PDfs if they are a major research output indicated on the page
    # by being downloadable by clicking a big red button
    parser.add_argument(
        "-c", "--cautious", action="store_true"
    )  # cautious flag. If present, asks whether you want to scrape the PDF
    # and whether the metadata guesses are correct

    # arguments only relevant in web_search mode
    parser.add_argument("--query")
    parser.add_argument("--site", default=NESTA_SITE_URL)
    parser.add_argument(
        "--use-subdirectories", action="store_true"
    )  # if present and site=NESTA_SITE_URL, search various subdirectories of the Nesta website in turn

    # arguments only relevant in given_urls or from_drive mode
    parser.add_argument("--urls", nargs="*")

    # arguments only relevant to from_drive mode
    parser.add_argument("--drive_type", type=DriveTypeEnum)
    parser.add_argument(
        "--all_drive", action="store_true"
    )  # all files flag. If present, attempt to ingest all PDF documents which are accessible in the Google Drive root directory.
    #
    parser.add_argument("--file_ids", nargs="*")
    # alternatively, you can specify a list of file_ids to ingest via the command line

    # arguments relevant to web_dump mode and from_csv mode
    parser.add_argument(
        "--start_index", type=int, default=0
    )  # the row of the relevant data file to start ingesting; everything prior to this will be ignored
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE
    )  # the number of webpages/rows to ingest at a time

    # parse command line arguments
    args = parser.parse_args()

    # translate command line arguments to mode
    mode: mode_type = "given_urls" if args.urls else mode_args_map.get(args.mode.value)

    # some other flags derived once the command line arguments are known
    # relevant to web_dump mode only
    download_button_pdf_only: bool = args.pdf and args.all
    split_documents = mode != "from_drive"  # if True, split documents into chunks before ingesting

    # relevant to from_drive mode
    drive_type: DriveTypeEnum = DriveTypeEnum(args.drive_type or "policy")
    pdf_parser: Literal[
        "unstructured", "openparse"
    ] = "openparse"  # use openparse for simple documents which may contain tables

    # check that the command line arguments are sensible
    error_instructions = "\n* give the command line argument '-m wd', '-m ws', '-m csv' or '-m drv' to signify one of the modes; OR\n* give a list of urls to go into 'given_urls' mode"  # noqa
    if not mode:
        raise Exception("No mode detected: You must either:" + error_instructions)
    elif args.urls and mode_args_map.get(args.mode):
        raise Exception(
            "Confusion in determining mode You must EITHER:" + error_instructions + "\nYou appear to have done both"
        )

    if mode == "web_search" and not args.query:
        raise Exception("You must provide a --query argument via the command line in web_search mode")
    if mode == "from_drive" and not args.file_ids and not args.urls and not args.all_drive:
        raise Exception(
            "You must EITHER provide a list of file_ids or URLs of the files you want to ingest, "
            "OR set --all-drive flag in from_drive mode"
        )

    # log command line arguments received
    info = ["", "\nIngestion settings as interpreted from command line arguments:"]
    info.append(f'mode: {mode} ({args.mode.value if args.mode else f"{len(args.urls)} urls provided"})')

    present_args = ["replace"]
    if mode == "web_dump":
        present_args += ["pdf", "all", "cautious", "start_index", "batch_size"]
    elif mode == "web_search":
        present_args += ["query", "site", "use_subdirectories"]
    elif mode == "given_urls":
        present_args += ["urls"]
    elif mode == "from_drive":
        present_args += ["drive_type", "all", "file_ids", "urls"]
    elif mode == "from_csv":
        present_args += ["start_index", "batch_size"]

    info += [f"{k}: {v}" for k, v in args.__dict__.items() if k in present_args]
    info.append("Refer to instructions if these are not correct\n")
    logger.info("\n".join(info))

    # execute ingestion depending on mode

    if mode == "web_dump":
        # if scraping/ingesting from entire Nesta website data dump

        metadata_df = pd.read_json(METADATA_PATH, lines=True)
        downloaded = metadata_df["_status_code"].apply(lambda val: val == 200)
        metadata_df = metadata_df[downloaded]
        n_rows = metadata_df.shape[0]

        for index in list(range(args.start_index, n_rows, args.batch_size)):
            df = metadata_df.iloc[index : (index + args.batch_size)]

            if args.pdf:
                pdfs_to_ingested_data(
                    df,
                    replace=args.replace,
                    download_button_pdf_only=download_button_pdf_only,
                    cautious=args.cautious,
                    split_documents=split_documents,
                )
            else:
                webpages_to_ingested_data(df=df, replace=args.replace, split_documents=split_documents)

    elif mode == "web_search":
        # if scraping from web

        if subdirectories:
            urls = [args.site + "/" + subdirectory for subdirectory in subdirectories]
        else:
            urls = [args.site]

        for url in urls:

            for start in list(
                range(0, 100, 10)
            ):  # the start parameter specifies which result set to return from Google Programmable Search;
                # 0 = first set of 10 results, 10 = the next set of 10 results, etc.
                logging.info(f"\nGoogle search result set url = {url}, start = {start}")

                results_returned = search_query_to_ingested_data(
                    args.query, url, start=start, replace=args.replace, split_documents=split_documents
                )
                if not results_returned:
                    break

    elif mode == "given_urls":

        scraped_data = []
        for url in args.urls:

            logger.info(f"\nScraping {url}")
            try:
                scraped_datum = scrape(url)
                scraped_datum["location"] = url
                scraped_data.append(scraped_datum)
            except Exception as e:
                logger.error(f"Error while trying to scrape {url}: {e}")

        scraped_data_to_ingested_data(scraped_data, replace=args.replace, split_documents=split_documents)

    elif mode == "from_drive":

        if args.file_ids:
            file_ids = args.file_ids
        elif args.urls:
            file_ids = [
                url.replace("https://drive.google.com/file/d/", "").replace("/view?usp=sharing", "")
                for url in args.urls
            ]
        else:
            file_ids = None

        ingest_from_drive(
            replace=args.replace,
            split_documents=split_documents,
            drive_type=drive_type,
            all=args.all_drive,
            file_ids=file_ids,
        )

    elif mode == "from_csv":
        # if ingesting data from a csv

        data = pd.read_csv(CSV_PATH)
        N_rows = data.shape[0]

        for start_index_ in range(args.start_index, N_rows, args.batch_size):

            ing.csv_rows_to_ingested_data(
                CSV_PATH,
                start_index_,
                args.batch_size,
                identifier="name",
                text_col=["Project Name (Asana)", "Research Question"],
                Chunk_func=lambda *args, **kwargs: chunk_to_Chunk(*args, **kwargs),
                chunk_presence_test=lambda *args, **kwargs: chunk_already_in_db(*args, **kwargs),
            )
