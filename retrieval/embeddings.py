from __future__ import annotations

import os

from typing import Coroutine
from typing import List
from typing import Union

from config import DEFAULT_EMBEDDINGS_MODEL
from config import USE_AZURE_EMBEDDINGS
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from openai import APIConnectionError
from openai import AsyncOpenAI
from openai import OpenAI
from openai import RateLimitError
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential


load_dotenv()


# define the sync and async embedding vector functions depending on whether we are using Azure or OpenAI

if USE_AZURE_EMBEDDINGS:

    embeddings_model = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDINGS_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_EMBEDDINGS_API_KEY"],
    )

    async def async_embedding_vector_(string: str) -> Coroutine:
        """Calculate the embedding vector of a string asynchronously using Azure OpenAI."""
        return await embeddings_model.aembed_query(string)

    def embedding_vector_(string: str) -> List[float]:
        """Calculate the embedding vector of a string synchronously using Azure OpenAI."""
        return embeddings_model.embed_query(string)

else:

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def async_embedding_vector_(string: str) -> Coroutine:
        """Calculate the embedding vector of a string asynchronously using OpenAI."""
        result = await async_client.embeddings.create(model=DEFAULT_EMBEDDINGS_MODEL, input=string)
        return result.data[0].embedding

    def embedding_vector_(string: str) -> List[float]:
        """Calculate the embedding vector of a string synchronously using OpenAI."""
        return client.embeddings.create(model=DEFAULT_EMBEDDINGS_MODEL, input=string).data[0].embedding


# add retry decorator for retrying if rate limit error or API connection error occurs
@retry(
    wait=wait_random_exponential(min=1, max=60),  # Exponential backoff with minimum 1 second and maximum 60 seconds
    stop=stop_after_attempt(5),  # Stop after 5 attempts
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError)
    ),  # Retry only on RateLimitError or APIConnectionError
)
async def async_embedding_vector(string: str) -> Coroutine:
    """Calculate the embedding vector of a string asynchronously with retry logic."""
    return await async_embedding_vector_(string)


@retry(
    wait=wait_random_exponential(min=1, max=60),  # Exponential backoff with minimum 1 second and maximum 60 seconds
    stop=stop_after_attempt(5),  # Stop after 5 attempts
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError)
    ),  # Retry only on RateLimitError or APIConnectionError
)
def embedding_vector(string: str) -> List[float]:
    """Calculate the embedding vector of a string synchronously with retry logic."""
    return embedding_vector_(string)


# calculate the embedding vector of a string, synchronously or asynchronously, with retry logic


def vector(string: str, async_: bool = False) -> Union[List[float], Coroutine]:
    """Calculate the embedding vector of string"""

    # intentionally not using the neater syntax documented by lanceDB which automatically calculates embeddings vectors
    # using model.VectorField() specified in the schema.
    # This is because there were issues getting a nested schema to work with this method.

    if async_:

        async def vector_() -> Coroutine:
            return await async_embedding_vector(string)

        return vector_()  # returns a coroutine which can be awaited

    else:

        return embedding_vector(string)
