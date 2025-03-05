from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from typing import List

from pydantic import BaseModel


class BaseDriveDoc(BaseModel):
    """A class for describing templates for documents on Google Drive (note: need not be a Google Doc)."""

    def __repr__(self) -> str:
        """Self-explanatory"""
        return "\n\t".join([f"{k}: {v}" for k, v in self.__dict__.items()])

    @abstractmethod
    @staticmethod
    def list(as_string: bool = False, to_csv: bool = False) -> List[BaseDriveDoc]:
        """List all the relevant drive docs"""
        pass

    def to_dict(self, drop: List[str] = None, order: List[str] = None) -> OrderedDict:
        """Return fields as an ordered dictionary, for example, for conversion into a row in a dataframe"""
        if order:
            return OrderedDict({k: self.__dict__[k] for k in order})
        else:
            return OrderedDict({k: v for k, v in self.__dict__.items() if k not in (drop or [])})
