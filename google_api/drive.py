from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
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
DEFAULT_SCOPES = ["https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/documents"]
READ_ONLY_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly"
]  # need this to read any document which nesta-brain-chatbot is not the owner of
# for example, downloading PDFs will not work if DEFAULT_SCOPES is used

# default Google Drive folder ID
DEFAULT_FOLDER_ID = "1WyMFiP4Q8NDILNXWCdmFJ7Tvlg37wL89"

SERVICE_ACCOUNT_FILE = "credentials.json"


def authenticate_service_account(scopes: List[str] = DEFAULT_SCOPES) -> Dict:
    """Authenticate using a service account."""
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
    return creds


def create_document(body_dict: Dict, **kwargs) -> str:
    """Create a Google Docs document."""

    doc = docs_service(**kwargs).documents().create(body=body_dict).execute()
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

    file_metadata = {"name": file_name, "mimeType": mimetype}

    media = MediaInMemoryUpload(string.strip().encode("utf-8"), mimetype=mimetype)
    if mimetype == "text/markdown":
        input("Check this has worked - before mimetype was 'text/plain'")

    # Create the file on Google Drive
    file = drive_service(creds=creds).files().create(body=file_metadata, media_body=media, fields="id").execute()

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


def docs_service(creds: Optional[Dict] = None) -> Resource:
    """Return a Google Docs service object."""
    creds = creds or authenticate_service_account()
    return build("docs", "v1", credentials=creds)


def download_pdf(
    file_id: str,
    service: Optional[Resource] = None,
    path: str = "google_api/downloaded.pdf",
    scopes: List[str] = READ_ONLY_SCOPES,
    silent: bool = False,
    **kwargs,
) -> None:
    """Download a PDF from Google Drive."""

    service = service or drive_service(scopes=scopes, **kwargs)
    request = service.files().get_media(fileId=file_id)
    with open(path, "wb") as file:
        file.write(request.execute())
    if not silent:
        logger.info(f"Downloaded PDF {file_id} to {path}")


def drive_service(creds: Optional[Dict] = None, scopes: List[str] = DEFAULT_SCOPES) -> Resource:
    """Return a Google Drive service object."""
    if not creds or scopes != DEFAULT_SCOPES:
        creds = authenticate_service_account(scopes=scopes)
    return build("drive", "v3", credentials=creds)


def get_document(
    document_id: str,
    creds: Optional[Dict] = None,
    service: Optional[Resource] = None,
    as_google_doc: bool = False,
    **kwargs,
) -> Union[Dict, GoogleDoc]:
    """Get a Google Doc"""

    logger.info(f"Getting {document_id} from Drive")

    service = service or docs_service(creds=creds)

    try:
        doc = service.documents().get(documentId=document_id, **kwargs).execute()

        if as_google_doc:
            doc = GoogleDoc(doc)

        return doc

    except HttpError as http_error:
        raise http_error


def get_file(
    document_id: str,
    creds: Optional[Dict] = None,
    service: Optional[Resource] = None,
    is_pdf: bool = False,
    silent: bool = False,
    **kwargs,
) -> Dict:
    """Get a file from Drive"""

    if not silent:
        logger.info(f'Getting file with ID "{document_id}" from Drive')

    service = service or drive_service(creds=creds, scopes=READ_ONLY_SCOPES if is_pdf else DEFAULT_SCOPES)
    file = service.files().get(fileId=document_id, **kwargs).execute()

    return file


def list_files(
    mimetype: Optional[str] = None, sub_folder_id: Optional[str] = None, read_only: bool = False
) -> List[Dict]:
    """List files in Google Drive."""

    subqueries = []
    scopes = DEFAULT_SCOPES

    if mimetype:
        subqueries.append(f"mimeType='{mimetype}'")

    if sub_folder_id:
        subqueries.append(f"'{sub_folder_id}' in parents")

    q = None
    if subqueries:
        q = " and ".join(subqueries)

    if mimetype == "application/pdf" or read_only:
        scopes = READ_ONLY_SCOPES

    result = drive_service(scopes=scopes).files().list(q=q).execute()
    return result["files"]


def move_file(
    file_id: str,
    to_folder: str = DEFAULT_FOLDER_ID,
    service: Optional[Resource] = None,
    silent: bool = False,
    **kwargs,
) -> None:
    """Move the document to a specific folder."""

    service = service or drive_service(**kwargs)
    file = get_file(file_id, service=service, fields="parents", silent=True)
    previous_parents = ",".join(file.get("parents", []))

    service.files().update(
        fileId=file_id, addParents=to_folder, removeParents=previous_parents, fields="id, parents"
    ).execute()

    if not silent:
        logger.info(f"File {file_id} moved to folder {to_folder}")

    return to_folder


if __name__ == "__main__":

    doc = get_document("foo", as_google_doc=True)
