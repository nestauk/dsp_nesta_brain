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
    """
    Defines the fields which a Chunk or chunk-like Record contains in the LanceDB database.
    Note that this need not be derived from a document, but could, for example represent a
    record in a table (if the record has a field which is wordy and may benefit from being
    subject to vector search)
    See the NestaBrain schema for a Chunk class which builds on this class and contains a `source`
    field to contain document metadata
    """  # noqa

    # chunk metadata
    text: str
    vector: Vector(model.ndims())
    order_index: Optional[int] = None
    time_added: Optional[datetime] = None
    # useful during deployment and not intended as metadata
    relevance_score: Optional[float] = None
    use_as_context: Optional[bool] = None

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
        metadata = self.metadata
        metadata.update({prop_name: getattr(self, prop_name) for prop_name in ["use_as_context"]})
        return self.to_LangchainDocument_(self.text, metadata, **kwargs)
