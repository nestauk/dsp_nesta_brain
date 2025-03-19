from __future__ import annotations

import sys

from typing import List

from google_api.drive import list_files
from google_api.drive_doc.base import BaseDriveDoc
from pydantic import Field


MODULE = sys.modules[__name__]
TEMPLATE_FOLDER_ID = "1u6m8rP5n0voWt2E0Y5Z-znMMBGgMc1Q8"

office_template_list = None


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

        if MODULE.office_template_list:
            templates = MODULE.office_template_list

        else:
            files = list_files(sub_folder_id=TEMPLATE_FOLDER_ID, read_only=True)

            templates = [OfficeTemplate(google_api_data=file) for file in files]
            MODULE.office_template_list = templates

        if kwargs:
            return OfficeTemplate.format_list(templates, **kwargs)

        else:
            return templates
