import asyncio
import os

from datetime import datetime
from typing import List
from typing import Optional

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
request_counter = {}
RPM_RATE_LIMIT = 10000
TPM_RATE_LIMIT = 5e6

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

db = lancedb.connect(DB_PATH)
chunk_table = db.open_table(chunk_table_name)


class RequestCounter(dict):
    """Stores information needed for the throttle"""

    def __init__(self) -> None:
        self["N_requests"] = []
        self["N_tokens"] = []

    @staticmethod
    def N_tokens(text: str) -> int:
        """Extimate the number of tokens in text"""
        return len(tiktoken.get_encoding(OPENAI_ENCODING).encode(text))

    @property
    def about_to_exceed_RPM_RATE_LIMIT(self) -> bool:
        """Determine whether the RPM rate limit will be exceeded if the batch of embeddings proceeds"""
        return sum(self["N_requests"]) >= RPM_RATE_LIMIT

    @property
    def about_to_exceed_TPM_RATE_LIMIT(self) -> bool:
        """Determine whether the TPM rate limit will be exceeded if the batch of embeddings proceeds"""
        return sum(self["N_tokens"]) >= TPM_RATE_LIMIT

    @property
    def hard_exceeds_RPM_RATE_LIMIT(self) -> bool:
        """Determine whether the RPM rate limit will be exceeded due to a single batch"""
        return self["N_requests"][-1] >= RPM_RATE_LIMIT

    @property
    def hard_exceeds_TPM_RATE_LIMIT(self) -> bool:
        """Determine whether the TPM rate limit will be exceeded due to a single batch"""
        return self["N_tokens"][-1] >= TPM_RATE_LIMIT

    @property
    def seconds_since_count_start(self) -> int:
        """Get seconds since the RequestCounter was set or reset"""
        if self.get("time"):
            return (datetime.now() - (self["time"])).seconds
        else:
            return 0

    def increment(self, texts: List[str]) -> None:
        """Increment the number of requests and number of tokens values"""
        self["N_requests"].append(len(texts))
        N_tokens_ = sum([self.N_tokens(text) for text in texts])
        self["N_tokens"].append(N_tokens_)

    def reset(self) -> None:
        """Reset the request counter"""
        self["time"] = datetime.now()
        self["N_requests"] = []
        self["N_tokens"] = []

    def update(self, texts: List[str]) -> int:
        """Keep a record of how many requests and tokens have been sent to the embeddings model"""

        seconds_since_count_start = request_counter.seconds_since_count_start

        if seconds_since_count_start >= 60:
            request_counter.reset()

        request_counter.increment(texts)

        return seconds_since_count_start


request_counter = RequestCounter()


async def throttle(texts: List[str]) -> None:
    """If embeddings model rate limits are exceeded, wait until sufficient time has passed"""
    # this won't work well for larger batch sizes, but unfortunately there isn't really time to troubleshoot and improve it
    # I still get API error messages back with batch_size >= 100 but that can't be due to hitting the rate limit
    # future users may want to improve on it

    seconds_since_count_start = request_counter.update(texts)

    if request_counter.hard_exceeds_RPM_RATE_LIMIT:
        raise Exception(f"You cannot ask for {RPM_RATE_LIMIT} or more requests to the embeddings model in one go")

    elif request_counter.hard_exceeds_TPM_RATE_LIMIT:
        raise Exception(
            f"You cannot ask for {TPM_RATE_LIMIT} or more tokens to be sent to the embeddings model in one go"
        )

    if seconds_since_count_start and seconds_since_count_start < 60:
        msg = None
        if request_counter.about_to_exceed_RPM_RATE_LIMIT:
            msg = "About to exceed OpenAI embeddings requests per minute rate limit ... sleeping for {sleep_time} seconds"
        elif request_counter.about_to_exceed_TPM_RATE_LIMIT:
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
