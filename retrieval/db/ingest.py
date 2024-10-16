import asyncio
import logging
import os

from datetime import datetime
from typing import Dict
from typing import List

import lancedb

from dotenv import load_dotenv
from dsp_nesta_brain import logger
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import CharacterTextSplitter
from openai import AsyncOpenAI
from retrieval.db.schema import Chunk
from retrieval.db.schema import Document as LanceDocument
from scraping.scrape import search_query_to_scraped_data


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

db = lancedb.connect("retrieval/db/ccid_demo_db")
document_table = db.open_table("document")
chunk_table = db.open_table("chunk")


def already_in_db(location: str) -> bool:
    """Determine whether chunks from a source document have already been added to the database"""
    results = chunk_table.search().where(f'source.location = "{location}"').to_list()
    return bool(results)


def ingest(documents: List[LangchainDocument]) -> None:
    """
    Split documents into chunks, derive embeddings for the chunks
    and insert Document and Chunk data (including embeddings) into the database
    """  # noqa

    async def to_Chunk(chunk: LangchainDocument) -> Chunk:
        # intentionally not using the neater syntax documented by lanceDB which automatically calculates embeddings vectors
        # using model.VectorField() specified in the schema.
        # This is because I had issues getting the nested schema to work with this method.
        async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        source = [doc for doc in lance_documents if doc.location == chunk.metadata["location"]][0]
        result = await async_client.embeddings.create(model="text-embedding-3-small", input=chunk.page_content)
        vector = result.data[0].embedding
        return Chunk(text=chunk.page_content, source=source, vector=vector)

    async def to_Chunks(chunks: List[LangchainDocument]) -> List[Chunk]:
        tasks = [asyncio.create_task(to_Chunk(chunk)) for chunk in chunks]
        return await asyncio.gather(*tasks)

    logging.getLogger("httpx").setLevel(logging.WARNING)

    not_already_in_db = [doc for doc in documents if not already_in_db(doc.metadata["location"])]
    N_not_in_db = len(not_already_in_db)
    if N_not_in_db != len(documents):
        logger.info(
            f"{len(documents) - N_not_in_db} documents were already in the database: {N_not_in_db} remain to be added"
        )
        documents = not_already_in_db

    if documents:

        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        docs_split = text_splitter.split_documents(documents)

        lance_documents = [LanceDocument(**doc.metadata) for doc in documents]

        logger.info(f"Fetching embeddings for {len(docs_split)} chunks ...")
        chunks = asyncio.run(to_Chunks(docs_split))

        # ====CAUTION====
        # document_table.add(lance_documents) introduces data redundancy in the database
        # and should be removed for later versions.
        # The source field in the chunk table does not link to a Document record.
        # If the title of a record in the document table is updated,
        # the source.title for the relevant chunk records remains the same
        # This is a recipe for mess!
        # I am keeping this in temporarily for purposes of experimentation
        document_table.add(lance_documents)
        chunk_table.add(chunks)

    logging.getLogger("httpx").setLevel(logging.INFO)


def search_query_to_ingested_data(query: str, site_url: str, **kwargs) -> bool:
    """Perform a search, scrape the webpages from the search results, and ingest the data"""

    scraped_data = search_query_to_scraped_data(query, site_url, **kwargs)

    docs = []
    for datum in scraped_data:
        doc = scraped_data_to_langchain_doc(datum)
        docs.append(doc)

    ingest(docs)

    return bool(scraped_data)


def scraped_data_to_langchain_doc(datum: Dict) -> LangchainDocument:
    """Convert a dict representing scraped data to a Langchain Document"""

    text = datum.get("text")
    metadata = {
        "location": datum.get("url"),
        "title": datum.get("title"),
        "date_pub": datum.get("date_pub"),
        "time_added": datetime.now(),
    }
    return LangchainDocument(page_content=text, metadata=metadata)


if __name__ == "__main__":

    query = "Centre for Collective Intelligence Design"
    site_url = "nesta.org.uk"
    subdirectories = sorted(
        ["toolkit", "team", "report", "project", "press-release", "jobs", "feature", "event", "blog"]
    )

    #  webpage_url = None #"https://www.nesta.org.uk/project/centre-collective-intelligence-design/"
    #    webpage_url = "https://www.nesta.org.uk/jobs/product-designer-centre-for-collective-intelligence-design-ccid/"

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


#    elif webpage_url:

#       scraped_data = [scrape(webpage_url)]
#      scraped_data['url'] = webpage_url
