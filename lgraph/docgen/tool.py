from typing import TYPE_CHECKING
from typing import Union

from dsp_nesta_brain import logger
from google_api.drive import get_document
from googleapiclient.errors import HttpError
from langchain_core.tools import tool


if TYPE_CHECKING:
    from google_api.google_doc import GoogleDoc


@tool
def get_template(model_output: str) -> Union[GoogleDoc, None]:
    """Get template"""

    if model_output != "NULL":

        doc_id = model_output

        try:
            google_doc = get_document(doc_id, as_google_doc=True)
            return google_doc

        except HttpError as http_error:
            logger.error(f'Error finding Google Doc with ID "{doc_id}": {http_error}')
