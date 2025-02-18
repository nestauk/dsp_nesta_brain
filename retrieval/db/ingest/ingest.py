from __future__ import annotations

import asyncio
import os

from datetime import datetime
from datetime import timedelta
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
    CHUNK_TABLE_NAME = "chunk"
elif PROJECT == "POLICY_ATLAS":
    Chunk = Activity
    CHUNK_TABLE_NAME = "activity"


CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100

OPENAI_ENCODING = "cl100k_base"

# OpenAI limits
request_counter = {}
RPM_RATE_LIMIT = 10000
TPM_RATE_LIMIT = int(5e6)

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

db = lancedb.connect(DB_PATH)
chunk_table = db.open_table(CHUNK_TABLE_NAME)


class RequestCounter(list):
    """Stores information needed for the throttle"""

    class RequestBatchData:
        """Stores information for each batch of texts sent to the embeddings model"""

        time: datetime
        N_requests: int
        N_tokens: int

        def __init__(self, texts: List[str]) -> None:
            self.time = datetime.now()
            self.N_requests = len(texts)
            self.N_tokens = len(tiktoken.get_encoding(OPENAI_ENCODING).encode("".join(texts)))

        @property
        def exceeds_RPM_rate_limit(self) -> bool:
            """Determine whether the RPM rate limit will be exceeded by request"""
            return self.N_requests >= RPM_RATE_LIMIT

        @property
        def exceeds_TPM_rate_limit(self) -> bool:
            """Determine whether the TPM rate limit will be exceeded by request"""
            return self.N_tokens >= TPM_RATE_LIMIT

    def exceeds_RPM_rate_limit(self, *args) -> bool:
        """Determine whether the RPM rate limit will be exceeded if the batch of embeddings proceeds"""
        return self.N_requests_since(*args) >= RPM_RATE_LIMIT

    def exceeds_TPM_rate_limit(self, *args) -> bool:
        """Determine whether the TPM rate limit will be exceeded if the batch of embeddings proceeds"""
        return self.N_tokens_since(*args) >= TPM_RATE_LIMIT

    def N_requests_since(self, since_time: datetime) -> int:
        """Calculate the number of requests since the time stated"""
        return sum([request_batch.N_requests for request_batch in self.request_batches_since(since_time)])

    def N_tokens_since(self, since_time: datetime) -> int:
        """Calculate the number of tokens since the time stated"""
        return sum([request_batch.N_tokens for request_batch in self.request_batches_since(since_time)])

    def report(self, since_time: datetime, **kwargs) -> None:
        """Log report on requests/tokens since time stated"""
        report_format = f"\nThrottle report: {self.N_requests_since(since_time)}/{RPM_RATE_LIMIT} requests, {self.N_tokens_since(since_time)}/{TPM_RATE_LIMIT} tokens since {since_time}: {{N_limits}} limit(s) breached: sleep time = {{sleep_time}}"  # noqa
        report = report_format.format(**kwargs)
        logger.info(report)

    def request_batches_since(self, since_time: datetime) -> List[RequestBatchData]:
        """Give a list of request batch data since the time stated"""
        return list(filter(lambda request_batch: request_batch.time >= since_time, self))

    def sleep_time(self, since_time: datetime) -> int:
        """Calculate the sleep time"""

        sleep_time = 0
        batch_at_start_of_exceedance = None
        for i in range(1, len(self)):
            sub_request_counter = RequestCounter(self[-(i + 1) :])
            if sub_request_counter.exceeds_RPM_rate_limit(since_time) or sub_request_counter.exceeds_TPM_rate_limit(
                since_time
            ):
                batch_at_start_of_exceedance = self[-(i + 1)]
                break

        if batch_at_start_of_exceedance:
            sleep_time = 60 - (datetime.now() - batch_at_start_of_exceedance.time).seconds

        return sleep_time


request_counter = RequestCounter()


async def throttle(request_counter: RequestCounter, texts: List[str]) -> None:
    """If embeddings model rate limits are exceeded, wait until sufficient time has passed, then proceed"""

    request_batch = RequestCounter.RequestBatchData(texts)
    request_counter.append(request_batch)

    error_message_format = "You cannot ask for {number} or more {rate_limit_name}s to the embeddings model in one go as this exceeds the {rate_limit} {rate_limit_name}s-per-minute rate limit"  # noqa
    if request_counter[-1].exceeds_RPM_rate_limit:
        raise Exception(
            error_message_format.format(
                number=request_counter[-1].N_requests, rate_limit=RPM_RATE_LIMIT, rate_limit_name="request"
            )
        )

    elif request_counter[-1].exceeds_TPM_rate_limit:
        raise Exception(
            error_message_format.format(
                number=request_counter[-1].N_tokens, rate_limit=TPM_RATE_LIMIT, rate_limit_name="token"
            )
        )

    else:
        sleep_time = 0
        since_time = datetime.now() - timedelta(seconds=60)
        exceeds_RPM_rate_limit = request_counter.exceeds_RPM_rate_limit(since_time)
        exceeds_TPM_rate_limit = request_counter.exceeds_TPM_rate_limit(since_time)

        if exceeds_RPM_rate_limit or exceeds_TPM_rate_limit:

            message_format = "About to exceed OpenAI embeddings {rate_limit_name}s per minute rate limit ... sleeping for {{sleep_time}} seconds"  # noqa
            if exceeds_RPM_rate_limit:
                message_format = message_format.format(rate_limit_name="request")
            elif exceeds_TPM_rate_limit:
                message_format = message_format.format(rate_limit_name="token")

            sleep_time = request_counter.sleep_time(since_time)

        request_counter.report(
            since_time, sleep_time=sleep_time, N_limits=sum([exceeds_RPM_rate_limit, exceeds_TPM_rate_limit])
        )

        if sleep_time:
            logger.info(message_format.format(sleep_time=sleep_time))

        await asyncio.sleep(
            sleep_time
        )  # do this even if sleep_time = 0 because the function needs to return a coroutine


def chunk_already_in_db(chunk: LangchainDocument, where_condition: Optional[str] = None) -> bool:
    """Determine whether identical chunks have already been added to the database, because PDFs may be duplicated across the site.
    Chunking strategy should have been the same.
    """  # noqa

    where_condition = where_condition or f'text == "{chunk.page_content}"'
    try:
        results = chunk_table.search().where(where_condition).limit(1).to_pydantic(Chunk)
    except Exception as e:
        error_message_format = "Error while trying to check whether activity {id} exists in database"
        logger.error(error_message_format.format(id=chunk.metadata.get("iati_identifier")))
        raise Exception(e)
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
