import asyncio
import os

from datetime import datetime
from typing import List
from typing import Optional
from typing import Union

import lancedb
import tiktoken

from config import DB_PATH
from config import DEFAULT_EMBEDDINGS_MODEL
from config import PROJECT
from dotenv import load_dotenv
from dsp_nesta_brain import logger
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import CharacterTextSplitter
from openai import AsyncOpenAI
from retrieval.db.schema.nesta_brain import Chunk as NestaBrainChunk
from retrieval.db.schema.policy_atlas import Activity


if PROJECT == "NESTA_BRAIN":
    Chunk = NestaBrainChunk
    chunk_table_name = "chunk"
elif PROJECT == "POLICY_ATLAS":
    Chunk = Activity
    chunk_table_name = "activity"


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
chunk_table = db.open_table(chunk_table_name)


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


def chunk_already_in_db(chunk: LangchainDocument, where_condition: Optional[str] = None) -> bool:
    """Determine whether identical chunks have already been added to the database, because PDFs may be duplicated across the site.
    Chunking strategy should have been the same.
    """  # noqa

    where_condition = where_condition or f'text == "{chunk.page_content}"'
    results = chunk_table.search().where(where_condition).limit(1).to_pydantic(Chunk)
    return results


async def chunk_to_Chunk(chunk: LangchainDocument, **kwargs) -> Chunk:
    """
    Convert a Langchain chunk (as returned from a text splitter) into an object
    of the Chunk class which can be ingested into the DB
    (including deriving an embedding for the Chunk)
    """  # noqa
    # intentionally not using the neater syntax documented by lanceDB which automatically calculates embeddings vectors
    # using model.VectorField() specified in the schema.
    # This is because I had issues getting a nested schema to work with this method.
    async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    result = await async_client.embeddings.create(model=DEFAULT_EMBEDDINGS_MODEL, input=chunk.page_content)
    vector = result.data[0].embedding
    return Chunk(text=chunk.page_content, vector=vector, **kwargs)


def split_documents(documents: List[LangchainDocument]) -> List[LangchainDocument]:
    """Split documents into chunks"""

    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs_split = text_splitter.split_documents(documents)

    if documents and not docs_split:
        raise Exception(f"Investigate why you have zero chunks for {len(documents)} documents")

    return docs_split
