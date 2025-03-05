from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

from dsp_nesta_brain import logger
from google_api.drive import create_document_in_folder_from_markdown
from google_api.drive import create_document_in_folder_from_string
from google_api.drive import get_document
from google_api.office_template import office_templates_as_dict
from googleapiclient.errors import HttpError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableParallel
from lgraph.docgen.prompt import apply_template_appendix
from llm.chain import history_aware_rag_chain
from llm.llm import default_llm as llm
from llm.message import InterimAIMessage
from llm.prompt import qa_system_prompt
from utils import yesno


# THIS IS AN ADAPTED VERSION OF PREVIOUS docgen branch CODE AND HASN'T BEEN TESTED YET


if TYPE_CHECKING:
    from lggraph.graph import State


MAX_ROUTER_RETRIES = 3


# -----nodes


def fetch_template(state: State) -> State:
    """Retrieve an office document template from Google Drive to help write a document"""

    file_ids = state["intermediate_outputs"].get("file_ids")
    if file_ids:

        file_id = file_ids[0]  # there should only be one, but test this

        try:
            google_doc = get_document(file_id, as_google_doc=True)
            state["intermediate_outputs"]["template"] = google_doc
            return state

        except HttpError as http_error:
            logger.error(f'Error finding Google Doc with ID "{file_id}": {http_error}')


def apply_template(state: State) -> State:
    """Apply a template retrieved from Drive to format the input request"""

    google_doc_template = state["intermediate_outputs"]["template"]

    prompt_template = qa_system_prompt + apply_template_appendix
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder("messages"),
        ]
    )
    state["messages"] = [msg for msg in state["messages"] if not isinstance(msg, InterimAIMessage)]

    chain = (
        RunnableParallel(
            template=(lambda: google_doc_template.text),
            context=(lambda x: x["context"]),
            messages=(lambda x: x["messages"]),
        )
        | prompt
        | llm
    )

    rag_chain = history_aware_rag_chain(chain)

    logger.info(f'Applying template "{google_doc_template.title}" to fulfil the request')
    response = rag_chain.invoke(state)

    state["messages"].append(response["answer"])

    return state


def upload_output(state: State) -> State:
    """
    Upload the content of the last message to Google Drive.

    Ideally the output should be in Markdown format
    """

    output = state["messages"][-1].content
    try:
        start_of_markdown = output.index(
            "#"
        )  # this is a bit weak – need a better way of detecting whether the string is Markdown
        markdown = output[start_of_markdown:]
        create_document_in_folder_from_markdown(markdown)
    except ValueError:
        logger.warning("Markdown not detected in LLM output – output will be uploaded to a text file")
        create_document_in_folder_from_string(output)

    return state


# -----conditional edges


def template_router(
    state: State,
) -> Literal["fetch_template", "call_default_chain", "decide_whether_needs_template"]:
    """Go the appropriate node, depending on whether an office template is needed"""

    file_ids = state["intermediate_outputs"].get("file_ids")

    if file_ids:
        file_id = file_ids[0]

        if file_id in office_templates_as_dict():  # replace office_templates_as_dict()
            return "fetch_template"

        else:
            invalid_template_message = "decide_whether_needs_template node returned an invalid template UID"
            previous_calls = [
                message
                for message in state["messages"]
                if getattr(message, "caller", None) == "decide_whether_needs_template"
            ]
            if len(previous_calls) <= MAX_ROUTER_RETRIES:
                logger.info(invalid_template_message + " – retrying")
                return "decide_whether_needs_template"
            else:
                logger.info(
                    invalid_template_message
                    + " - however, the maximum number of retries for decide_whether_needs_template node have already been met. Proceeding without using template"  # noqa
                )

    return "call_default_chain"


def upload_router(state: State) -> Literal["upload_output", "wash_up"]:
    """Go the appropriate node, depending on whether the output should be uploaded to Google Drive"""

    if yesno("Upload the output?"):  # temporary decision process to check it works
        # – obviously users ultimately won't be interacting with this via the command line
        return "upload_output"

    return "wash_up"
