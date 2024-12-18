from __future__ import annotations

import os

from config import DEFAULT_MODEL
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

default_llm = ChatOpenAI(
    temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=DEFAULT_MODEL, streaming=True
)
