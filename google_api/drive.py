from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

from dsp_nesta_brain import logger
from google.oauth2.service_account import Credentials
from google_api.google_doc import GoogleDoc
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaInMemoryUpload


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


def create_document_in_folder_from_string(
    string: str,
    file_name: str = "tmp.txt",
    mimetype: str = "text/plain",
    creds: Optional[Dict] = None,
    silent: bool = False,
    **kwargs,
) -> Tuple[str]:
    """Create a Google Docs document from a string in a specific folder."""

    creds = creds or authenticate_service_account()
    service = build("drive", "v3", credentials=creds)

    file_metadata = {"name": file_name, "mimeType": mimetype}

    media = MediaInMemoryUpload(string.strip().encode("utf-8"), mimetype=mimetype)
    if mimetype == "text/markdown":
        input("Check this has worked - before mimetype was 'text/plain'")

    # Create the file on Google Drive
    file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()

    folder = move_file(file["id"], silent=silent, **kwargs)

    if not silent:
        logger.info(f"File {file['id']} created in folder {folder}")

    return file["id"], folder


def create_document_in_folder_from_markdown(markdown_string: str, file_name: str = "tmp.md", **kwargs) -> None:
    """Create a Google Docs document from a markdown string in a specific folder."""

    if not file_name[-3:] == ".md":
        raise Exception(f'file_name "{file_name}" in create_document_from_markdown did not have .md extension')

    file_id, folder = create_document_in_folder_from_string(
        markdown_string, file_name=file_name, mimetype="text/markdown", **kwargs
    )

    logger.info(f"Markdown file {file_id} created in folder {folder}")


def create_document_in_folder(*args, creds: Optional[Dict] = None, **kwargs) -> None:
    """Create a Google Docs document in a specific folder."""

    creds = creds or authenticate_service_account()
    document_id = create_document(*args, creds=creds)
    move_file(document_id, creds=creds, **kwargs)


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


def get_file(
    document_id: str, creds: Optional[Dict] = None, service: Optional[Resource] = None, silent: bool = False, **kwargs
) -> Dict:
    """Get a file from Drive"""

    if not silent:
        logger.info(f'Getting file with ID "{document_id}" from Drive')

    if not service:
        creds = creds or authenticate_service_account()
        service = build("drive", "v3", credentials=creds)

    file = service.files().get(fileId=document_id, **kwargs).execute()

    return file


def move_file(
    file_id: str, to_folder: str = DEFAULT_FOLDER_ID, creds: Optional[Dict] = None, silent: bool = False
) -> None:
    """Move the document to a specific folder."""

    creds = creds or authenticate_service_account()
    drive_service = build("drive", "v3", credentials=creds)

    file = get_file(file_id, service=drive_service, fields="parents", silent=True)
    previous_parents = ",".join(file.get("parents", []))

    drive_service.files().update(
        fileId=file_id, addParents=to_folder, removeParents=previous_parents, fields="id, parents"
    ).execute()

    if not silent:
        logger.info(f"File {file_id} moved to folder {to_folder}")

    return to_folder


if __name__ == "__main__":

    doc = get_document("foo", as_google_doc=True)
