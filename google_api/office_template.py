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

    def __repr__(self) -> str:
        """Self-explanatory"""
        format_ = "\n\tUID: {UID}\n\ttitle: {title}\n\tpurpose: {purpose}\n"
        return format_.format(**{k: getattr(self, k) for k in self.__class__.dict(self) if k in format_})


office_templates = [
    OfficeTemplate(
        UID="1J-cDWmCBf2EWtthhvTIvoF6enFf2EipoEiZUzoTRxxo",
        title="Project Proposal Template",
        # purpose="Help staff write proposals at the Opportunity and Scoping phases",
        purpose="Help staff structure project proposals",
    ),
    OfficeTemplate(
        UID="PJUPDT",
        title="Project Updates Guidance",
        purpose="Help staff write webpages informing the public about project updates",
    ),
]


def office_templates_as_dict() -> Dict:
    """Convert office templates list into a dict for easy reference"""
    return {template.UID: template for template in office_templates}
