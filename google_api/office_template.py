from __future__ import annotations

from typing import List

from google_api.base import BaseDriveDoc
from google_api.drive import list_files
from pydantic import Field


TEMPLATE_FOLDER_ID = "1u6m8rP5n0voWt2E0Y5Z-znMMBGgMc1Q8"


class OfficeTemplate(BaseDriveDoc):
    """A class for describing templates for proposals, project updates, etc."""

    title: str = (
        Field(
            ...,
            description="The template title",
        ),
    )
    #  purpose: str = (
    #     Field(
    #        ...,
    #       description="What the template is for",
    #  ),
    # )

    @staticmethod
    def list(**kwargs) -> List[OfficeTemplate]:
        """Get all list of all the Office templates available and convert to the OfficeTemplate class"""

        files = list_files(sub_folder_id=TEMPLATE_FOLDER_ID, read_only=True)

        templates = [OfficeTemplate(google_api_data=file) for file in files]

        if kwargs:
            return OfficeTemplate.format_list(templates, **kwargs)

        else:
            return templates

    @property
    def prompt_template(self) -> str:
        """Return a string for a prompt template to use in a LangGraph node relating to the document"""

        return """
            You are a helpful assistant and an expert on the internal administration, personnel and projects of the innovation agency Nesta.

            Look at the following list of office document templates available to you to help staff write documents: {list}

            Look at the request below and decide whether the one of the document templates in the list is needed to fulfil it.

            If so, select the appropriate template. As your response, give only the file_id of the template you have selected.

            Otherwise, respond "NULL".

            Request:
            {{input}}
            """  # noqa
