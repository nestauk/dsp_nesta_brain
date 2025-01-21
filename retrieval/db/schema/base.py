from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import Dict
from typing import Optional

from config import DEFAULT_EMBEDDINGS_MODEL
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel
from lancedb.pydantic import Vector
from langchain.docstore.document import Document as LangchainDocument


model = get_registry().get("openai").create(name=DEFAULT_EMBEDDINGS_MODEL)


class BaseChunk(LanceModel):
    """Defines the fields which a Chunk or chunk-like Record contains in the LanceDB database"""

    text: str
    vector: Vector(model.ndims())
    order_index: Optional[
        int
    ] = None  # only optional because the very earliest entries in the Nesta Brain project don't have it
    time_added: Optional[datetime] = None

    @abstractmethod
    # @property  #should have a metadata method which uses @property decorator;
    # can't use both @abstractmethod and @property decorators here
    def metadata(self) -> Dict:
        """Abstract method for metadata"""
        pass

    @staticmethod
    def to_LangchainDocument_(text: str, metadata: Dict, enumeration_index: Optional[int] = None) -> LangchainDocument:
        """
        Convert text into a Langchain Document

        Having this as a separate method to to_LangchainDocument is useful when merging chunks
        """
        if enumeration_index:
            text = f"[{enumeration_index}] {text}"
        return LangchainDocument(page_content=text, metadata=metadata)

    def to_LangchainDocument(self, **kwargs) -> LangchainDocument:
        """Convert a Chunk into a Langchain Document"""
        return self.to_LangchainDocument_(self.text, self.metadata, **kwargs)
