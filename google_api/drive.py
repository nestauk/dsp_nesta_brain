from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import Optional
from typing import Union

from dsp_nesta_brain import logger
from google.oauth2.service_account import Credentials
from google_api.google_doc import GoogleDoc
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


if TYPE_CHECKING:
    from googleapiclient.discovery import Resource


# Define the scopes for both Google Drive and Google Docs
SCOPES = ["https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/documents"]


# default Google Drive folder ID
DEFAULT_FOLDER_ID = "1WyMFiP4Q8NDILNXWCdmFJ7Tvlg37wL89"

SERVICE_ACCOUNT_FILE = "credentials.json"


def authenticate_service_account() -> Dict:
    """Authenticate using a service account."""
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return creds


def create_document(body_dict: Dict, creds: Optional[Dict] = None) -> str:
    """Create a Google Docs document."""

    creds = creds or authenticate_service_account()
    docs_service = build("docs", "v1", credentials=creds)

    doc = docs_service.documents().create(body=body_dict).execute()
    document_id = doc["documentId"]
    logger.info(f"Document {document_id} created: https://docs.google.com/document/d/{document_id}/edit")

    return document_id


def create_document_in_folder(*args, creds: Optional[Dict] = None, **kwargs) -> None:
    """Create a Google Docs document in a specific folder."""

    creds = creds or authenticate_service_account()
    document_id = create_document(*args, creds=creds)
    move_document(document_id, creds=creds, **kwargs)


def get_document(
    document_id: str,
    creds: Optional[Dict] = None,
    service: Optional[Resource] = None,
    as_google_doc: bool = False,
    **kwargs,
) -> Union[Dict, GoogleDoc]:
    """Get a Google Doc"""

    logger.info(f"Getting {document_id} from Drive")

    if not service:
        creds = creds or authenticate_service_account()
        service = build("docs", "v1", credentials=creds)

    try:
        doc = service.documents().get(documentId=document_id, **kwargs).execute()

        if as_google_doc:
            doc = GoogleDoc(doc)

        return doc

    except HttpError as http_error:
        raise http_error


def get_file(document_id: str, creds: Optional[Dict] = None, service: Optional[Resource] = None, **kwargs) -> Dict:
    """Get a file from Drive"""

    logger.info(f'Getting with ID "{document_id}" from Drive')

    if not service:
        creds = creds or authenticate_service_account()
        service = build("drive", "v3", credentials=creds)

    file = service.files().get(fileId=document_id, **kwargs).execute()

    return file


def move_document(document_id: str, to_folder: str = DEFAULT_FOLDER_ID, creds: Optional[Dict] = None) -> None:
    """Move the document to a specific folder."""

    creds = creds or authenticate_service_account()
    drive_service = build("drive", "v3", credentials=creds)

    file = get_file(document_id, service=drive_service, fields="parents")
    previous_parents = ",".join(file.get("parents", []))

    drive_service.files().update(
        fileId=document_id, addParents=to_folder, removeParents=previous_parents, fields="id, parents"
    ).execute()

    logger.info(f"Document {document_id} moved to folder {to_folder}")


if __name__ == "__main__":

    doc = get_document("foo", as_google_doc=True)


#  create_document_in_folder({"title": "Fake Template"})
