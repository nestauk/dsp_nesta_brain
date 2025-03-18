from __future__ import annotations

import os

from config import AZURE_API_VERSION
from config import AZURE_MODEL
from config import DEFAULT_MODEL
from config import USE_AZURE_LLM
from dotenv import load_dotenv


load_dotenv()

if USE_AZURE_LLM:

    from langchain_openai import AzureChatOpenAI

    default_llm = AzureChatOpenAI(
        azure_deployment=AZURE_MODEL,
        api_version=AZURE_API_VERSION,
        temperature=0,
        #  max_tokens=200,   #may need to experiment with this if you get errors back from the API
        # timeout=None,
        max_retries=2,
    )


else:

    from langchain_openai import ChatOpenAI

    default_llm = ChatOpenAI(
        temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=DEFAULT_MODEL, streaming=True
    )
