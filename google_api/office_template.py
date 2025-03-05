from __future__ import annotations

from typing import Dict

from pydantic import BaseModel
from pydantic import Field


class OfficeTemplate(BaseModel):
    """A class for describing templates for proposals, project updates, etc."""

    UID: str = (
        Field(
            ...,
            description="A unique identifier",
        ),
    )
    title: str = (
        Field(
            ...,
            description="The template title",
        ),
    )
    purpose: str = (
        Field(
            ...,
            description="What the template is for",
        ),
    )

    @staticmethod
    def list() -> Dict[str, OfficeTemplate]:
        """Get all list of all the Office templates available and convert to the OfficeTemplate class"""

        pass
