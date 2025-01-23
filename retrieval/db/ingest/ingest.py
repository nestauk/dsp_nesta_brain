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
TPM_RATE_LIMIT = int(5e6)

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

db = lancedb.connect(DB_PATH)
chunk_table = db.open_table(chunk_table_name)


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
        return request_counter.N_requests_since(*args) >= RPM_RATE_LIMIT

    def exceeds_TPM_rate_limit(self, *args) -> bool:
        """Determine whether the TPM rate limit will be exceeded if the batch of embeddings proceeds"""
        return request_counter.N_tokens_since(*args) >= TPM_RATE_LIMIT

    def N_requests_since(self, since_time: datetime) -> int:
        """Calculate the number of requests since the time stated"""
        return sum([request_batch.N_requests for request_batch in self.request_batches_since(since_time)])

    def N_tokens_since(self, since_time: datetime) -> int:
        """Calculate the number of tokens since the time stated"""
        return sum([request_batch.N_tokens for request_batch in self.request_batches_since(since_time)])

    def report(self, since_time: datetime, **kwargs) -> None:
        """Log report on requests/tokens since time stated"""
        report_format = f"\nThrottle report: {self.N_requests_since(since_time)}/{RPM_RATE_LIMIT} requests, {self.N_tokens_since(since_time)}/{TPM_RATE_LIMIT} tokens since {since_time}: {{N_limits}} limits breached: sleep time = {{sleep_time}}"  # noqa
        report = report_format.format(**kwargs)
        logger.info(report)

    def request_batches_since(self, since_time: datetime) -> List[RequestBatchData]:
        """Give a list of request batch data since the time stated"""
        return list(filter(lambda request_batch: request_batch.time >= since_time, self))

    def sleep_time(self, since_time: datetime) -> int:
        """Calculate the sleep time"""

        sleep_time = 0
        exceedance_request_batch = None
        for i in range(1, len(self)):
            sub_request_counter = RequestCounter(self[-(i + 1) :])
            if sub_request_counter.exceeds_RPM_rate_limit(since_time) or sub_request_counter.exceeds_TPM_rate_limit(
                since_time
            ):
                exceedance_request_batch = self[-(i + 1)]
                break

        if exceedance_request_batch:
            sleep_time = 60 - (datetime.now() - exceedance_request_batch.time).seconds

        return sleep_time


request_counter = RequestCounter()


async def throttle(request_counter: RequestCounter, texts: List[str]) -> None:
    """If embeddings model rate limits are exceeded, wait until sufficient time has passed"""
    # this won't work well for larger batch sizes, but unfortunately there isn't really time to troubleshoot and improve it
    # I still get API error messages back with batch_size >= 100 but that can't be due to hitting the rate limit
    # future users may want to improve on it

    request_batch = RequestCounter.RequestBatchData(texts)
    request_counter.append(request_batch)

    if request_counter[-1].exceeds_RPM_rate_limit:
        raise Exception(f"You cannot ask for {RPM_RATE_LIMIT} or more requests to the embeddings model in one go")

    elif request_counter[-1].exceeds_TPM_rate_limit:
        raise Exception(
            f"You cannot ask for {TPM_RATE_LIMIT} or more tokens to be sent to the embeddings model in one go"
        )

    else:
        sleep_time = 0
        since_time = datetime.now() - timedelta(seconds=60)
        exceeds_RPM_rate_limit = request_counter.exceeds_RPM_rate_limit(since_time)
        exceeds_TPM_rate_limit = request_counter.exceeds_TPM_rate_limit(since_time)

        if exceeds_RPM_rate_limit or exceeds_TPM_rate_limit:

            message_format = "About to exceed OpenAI embeddings {rate_limit} per minute rate limit ... sleeping for {{sleep_time}} seconds"  # noqa
            if exceeds_RPM_rate_limit:
                message_format = message_format.format(rate_limit="requests")
            elif exceeds_TPM_rate_limit:
                message_format = message_format.format(rate_limit="tokens")

            sleep_time = request_counter.sleep_time
            logger.info(message_format.format(sleep_time=sleep_time))

        request_counter.report(
            since_time, sleep_time=sleep_time, N_limits=sum([exceeds_RPM_rate_limit, exceeds_TPM_rate_limit])
        )

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
