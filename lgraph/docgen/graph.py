from __future__ import annotations

import importlib

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

from dsp_nesta_brain import logger
from google_api.drive import create_document_in_folder_from_markdown
from google_api.drive import create_document_in_folder_from_string
from google_api.drive import get_document
from google_api.office_template import office_templates_as_dict
from googleapiclient.errors import HttpError
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from lgraph.docgen.prompt import apply_template_appendix
from lgraph.docgen.prompt import apply_template_prompt
from lgraph.docgen.prompt import needs_template_prompt
from llm.llm import default_llm as llm
from llm.message import CustomAIMessage
from llm.prompt import qa_system_prompt
from retrieval.retrieve import RetrieverInput as InputState
from utils import yesno


# from utils import yesno


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


MAX_ROUTER_RETRIES = 3
LAST_CHAT_GRAPH_NODE_NAME = "currentness_comment"


class OverallState(InputState):
    """Defines the state used internally and returned by the final node"""

    outputs: List[Any]


class InterimAIMessage(AIMessage):
    """Messages which are not wanted in the final chat history"""

    caller: Optional[str] = None  # the name of the node function where the message was created

    def __init__(self, message: AIMessage, **kwargs) -> None:
        super().__init__(content=message.content, **kwargs)


# -----nodes


def decide_whether_needs_template(state: InputState) -> OverallState:
    """Decide whether an office template is needed"""

    chain = RunnablePassthrough.assign(input=(lambda x: x["messages"][-1])) | needs_template_prompt | llm

    message = chain.invoke(state)

    state["messages"].append(InterimAIMessage(message, caller="decide_whether_needs_template"))

    return state


def fetch_template(state: OverallState) -> OverallState:
    """Retrieve an office document template from Google Drive to help write a document"""

    if state["messages"][-1].content != "NULL":

        doc_id = state["messages"][-1].content

        try:
            google_doc = get_document(doc_id, as_google_doc=True)
            if "outputs" not in state:
                state["outputs"] = []
            state["outputs"].append(google_doc)
            return state

        except HttpError as http_error:
            logger.error(f'Error finding Google Doc with ID "{doc_id}": {http_error}')


def apply_template(state: OverallState) -> OverallState:
    """Apply a template retrieved from Drive to format the input request"""

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
            template=(lambda state: state["outputs"][-1].text),
            context=(lambda x: x["context"]),
            messages=(lambda x: x["messages"]),
        )
        | prompt
        | llm
    )

    rag_chain = importlib.import_module("llm.chain").history_aware_rag_chain(chain)

    logger.info(f'Applying template "{state["outputs"][-1].title}" to fulfil the request')
    response = rag_chain.invoke(state)

    state["messages"].append(response["answer"])

    return state


def apply_template_old(state: OverallState) -> OverallState:
    """Apply a template retrieved from Drive to format the input request"""

    chain = (
        RunnableParallel(
            input=(lambda state: state["messages"][-1].content), template=(lambda state: state["outputs"][-1].text)
        )
        | apply_template_prompt
        | llm
    )

    logger.info(f'Applying template "{state["outputs"][-1].title}" to fulfil the request')
    message = chain.invoke(state)

    state["messages"].append(message)

    return state


def upload_output(state: OverallState) -> OverallState:
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


def filter_messages(state: OverallState) -> Dict:
    """
    Remove interim AI messages

    Needed when the chat history is about to be used (if interim messages have been appended)
    """
    # see https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/#manually-deleting-messages
    # for an explanation of the syntax
    # this works with MessagesState's default reducer
    # note that if the reducer for the messages key is changed, this may no longer work
    return {
        "messages": [
            RemoveMessage(id=message.id) for message in state["messages"] if isinstance(message, InterimAIMessage)
        ]
    }


# -----conditional edges


def template_router(
    state: OverallState,
) -> Literal["fetch_template", "call_default_chain", "decide_whether_needs_template"]:
    """Go the appropriate node, depending on whether an office template is needed"""

    office_template_UID = state["messages"][-1].content

    if office_template_UID != "NULL":
        if office_template_UID in office_templates_as_dict():
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


def upload_router(state: OverallState) -> Literal["upload_output", "wash_up"]:
    """Go the appropriate node, depending on whether the output should be uploaded to Google Drive"""

    if yesno("Upload the output?"):  # temporary decision process to check it works
        # – obviously users ultimately won't be interacting with this via the command line
        return "upload_output"

    return "wash_up"


# -----graph


def create_chat_graph(
    allow_document_generation: bool = False, **kwargs
) -> CompiledStateGraph:  # doing it as a function to avoid circular imports
    """Compile and return a graph to assist with chat"""

    rag_chain = importlib.import_module("llm.chain").history_aware_rag_chain(**kwargs)  # avoiding circular import

    def call_default_chain(
        state: OverallState,
    ) -> OverallState:  # function defined here to avoid circular import

        response = rag_chain.invoke(state)
        state["messages"].append(CustomAIMessage(response))

        return state

    builder = StateGraph(OverallState)

    builder.add_node("call_default_chain", call_default_chain)
    if allow_document_generation:
        builder.add_node("filter_messages", filter_messages)
        builder.add_node("decide_whether_needs_template", decide_whether_needs_template)
        # builder.add_node("fetch_template", fetch_template)
        builder.add_node("fetch_template", fetch_template)
        builder.add_node("apply_template", apply_template)
        builder.add_node("upload_output", upload_output)
        builder.add_node(
            "wash_up", filter_messages
        )  # this (redundantly) uses the filter_messages function for the moment, but may do something else at a later stage

    if allow_document_generation:
        builder.add_edge(START, "decide_whether_needs_template")
        builder.add_conditional_edges("decide_whether_needs_template", template_router)
        builder.add_edge("fetch_template", "filter_messages")
        builder.add_edge("filter_messages", "apply_template")
        builder.add_conditional_edges("apply_template", upload_router)
        builder.add_edge("upload_output", "wash_up")
        builder.add_edge("call_default_chain", "wash_up")
        builder.add_edge("wash_up", END)

    else:
        builder.add_edge(START, "call_default_chain")
        builder.add_edge("call_default_chain", END)

    return builder.compile()


if __name__ == "__main__":

    graph = create_chat_graph()

    if True:
        graph.get_graph().draw_mermaid_png(output_file_path="lgraph/mermaid.png")

    else:
        input = "Help me write a project proposal for a Collective Intelligence project"
        # input = "What does a lemon look like?"
        input = {"messages": [HumanMessage(content=input)], "filter_condition": "", "merge": True}
        res = graph.invoke(input)

        for message in res["messages"]:
            message.pretty_print()
