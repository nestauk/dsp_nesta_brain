from __future__ import annotations

import os

from typing import Callable
from typing import List
from typing import Union

from config import DEFAULT_EMBEDDINGS_MODEL
from config import USE_AZURE_EMBEDDINGS
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from openai import AsyncOpenAI
from openai import OpenAI


load_dotenv()

if USE_AZURE_EMBEDDINGS:
    embeddings_model = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDINGS_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_EMBEDDINGS_API_KEY"],
    )
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def vector(string: str, async_: bool = False) -> Union[List[float], Callable]:
    """Calculate the embedding vector of string"""

    # intentionally not using the neater syntax documented by lanceDB which automatically calculates embeddings vectors
    # using model.VectorField() specified in the schema.
    # This is because I had issues getting a nested schema to work with this method.

    if USE_AZURE_EMBEDDINGS:

        if async_:

            async def vector_() -> Callable:
                return await embeddings_model.aembed_query(string)

            return vector_()  # returns a coroutine which can be awaited

        else:
            return embeddings_model.embed_query(string)

    else:

        if async_:

            async def vector_() -> Callable:
                result = await async_client.embeddings.create(model=DEFAULT_EMBEDDINGS_MODEL, input=string)
                return result.data[0].embedding

            return vector_()  # returns a coroutine which can be awaited

        else:
            return client.embeddings.create(model=DEFAULT_EMBEDDINGS_MODEL, input=string).data[0].embedding
