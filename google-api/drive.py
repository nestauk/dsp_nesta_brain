from typing import Dict
from typing import Optional

from dsp_nesta_brain import logger
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


# Define the scopes for both Google Drive and Google Docs
SCOPES = ["https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/documents"]


# default Google Drive folder ID
DEFAULT_FOLDER_ID = "1WyMFiP4Q8NDILNXWCdmFJ7Tvlg37wL89"

SERVICE_ACCOUNT_FILE = "redentials.json"


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


def move_document(document_id: str, folder: str = DEFAULT_FOLDER_ID, creds: Optional[Dict] = None) -> None:
    """Move the document to a specific folder."""

    creds = creds or authenticate_service_account()
    drive_service = build("drive", "v3", credentials=creds)

    file = drive_service.files().get(fileId=document_id, fields="parents").execute()
    previous_parents = ",".join(file.get("parents", []))

    drive_service.files().update(
        fileId=document_id, addParents=folder, removeParents=previous_parents, fields="id, parents"
    ).execute()

    logger.info(f"Document {document_id} moved to folder {folder}")


def create_document_in_folder(*args, creds: Optional[Dict] = None, **kwargs) -> None:
    """Create a Google Docs document in a specific folder."""

    creds = creds or authenticate_service_account()
    document_id = create_document(*args, creds=creds)
    move_document(document_id, creds=creds, **kwargs)


if __name__ == "__main__":

    create_document_in_folder({"title": "Helen Test Doc"})
