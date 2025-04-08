from __future__ import annotations

import asyncio
import importlib
import re

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Dict
from typing import Literal
from typing import Type

from config import ALLOW_POLICY_DOCS
from config import DEFAULT_START_YEAR
from dsp_nesta_brain import logger
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.types import StreamWriter
from lgraph.prompt import currentness_comment_prompt
if ALLOW_POLICY_DOCS:
    from lgraph.prompt import needs_policy_prompt
from lgraph.prompt import personnel_prompt
from lgraph.prompt import year_constraint_prompt
from llm.llm import default_llm as llm
from llm.message import CustomAIMessage
from llm.prompt import qa_prompt
from llm.prompt import qa_verbatim_prompt
from llm.tool import year_range
from retrieval.retrieve import RetrieverInput


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


DEFAULT_FROM_YEAR_FILTER_CONDITION = f"source.date_pub >= to_timestamp('{DEFAULT_START_YEAR}-01-01')"

graph_options_type: Type = Literal["retrieval", "chat", "combined"]


class State(RetrieverInput):
    """State class for the graph"""

    intermediate_outputs: Dict
    raw_response: Dict


def append_filter_condition(state: State, new_filter_condition: str) -> State:

    """
    Replace DEFAULT_FROM_YEAR_FILTER_CONDITION if new_filter_condition is attempting to set a new from_year.
    Otherwise, make sure 'and' is appended before the new filter condition if a filter condition already exists
    """  # noqa

    if (
        state["filter_condition"]
        and "source.date_pub >=" in new_filter_condition
        and DEFAULT_FROM_YEAR_FILTER_CONDITION in state["filter_condition"]
    ):

        state["filter_condition"] = state["filter_condition"].replace(
            DEFAULT_FROM_YEAR_FILTER_CONDITION, new_filter_condition
        )

    else:
        if state["filter_condition"]:
            state["filter_condition"] += " and "
        state["filter_condition"] += new_filter_condition

    return state


# -------NODES
# NB: create async versions of any functions used in a graph which is invoked asynchronously

def decide_if_person_page(state: State) -> State:
    """Decide if the retrieval should be limited to person pages only; add appropriate filter_condition to the state if so"""
    if "source.contentType" not in state["filter_condition"]:  # if the user has explicitly set a filter condition via the UI, use that one and ignore the node
        chain = personnel_prompt | llm
        response = chain.invoke(state["input"])
        if response.content == "YES":
            state = append_filter_condition(state, "source.contentType = 'person page'")
    return state


def decide_if_need_time_constraint(state: State) -> State:
    """Decide if the publication date of retrieved documents should be constrained by a year range"""

    from_year_widget_has_been_set = (
        "source.date_pub" in state["filter_condition"]
        and DEFAULT_FROM_YEAR_FILTER_CONDITION not in state["filter_condition"]
    )
    if (
        not from_year_widget_has_been_set
    ):  # if the user has explicitly set a filter condition via the UI, use that one and ignore the node
        llm_with_tool = llm.bind_tools(  # remember if binding tools then llm must be a ChatModel
            [year_range],
            tool_choice="year_range",
        )
        chain = year_constraint_prompt | llm_with_tool
        response = chain.invoke(state["input"])
        response = response.tool_calls[0]["args"]
        if response.get("start_year") and response.get("end_year"):
            filter_condition = year_range(**response).to_filter_condition()
            state = append_filter_condition(state, filter_condition)
    #   print("within 'decide_if_need_time_constraint': ", state, response)
    return state


def interpret_policy_decision(message: AIMessage, state: State) -> State:

    """
    Interpret which policy documents are needed based on the user's response, and append the
    retrieval filter condition accordingly
    """  # noqa

    if message.content != "NULL":

        file_ids = message.content
        file_ids = [file_id.strip() for file_id in file_ids.split(",")]
        file_ids_are_right_format = all(re.search(r"^[A-Za-z0-9\-_]+$", file_id) for file_id in file_ids)

        if file_ids_are_right_format:
            logger.info(f"Policy document IDs identified: {file_ids}")
            state["intermediate_outputs"]["policy_file_ids"] = file_ids
            filter_condition = "(" + " or ".join([f'source.location LIKE "%{file_id}"' for file_id in file_ids]) + ")"
            state["use_hybrid_search"] = False
            # filter_condition = f'(source.drive_type == "policy" or source.location LIKE "%{file_id}")'
            state = append_filter_condition(state, filter_condition)
        else:
            logger.warning(
                f"Policy document IDs did not seem to be in the correct format. Message content: {message.content}"
            )

    return state

async def async_decide_whether_needs_policy(state: State) -> State:
    """Decide whether a policy document is needed"""

    chain = RunnablePassthrough.assign(input=(lambda x: x["messages"][-1])) | needs_policy_prompt | llm
    message = await chain.ainvoke(state)
    state = interpret_policy_decision(message, state)
    return state

def decide_whether_needs_policy(state: State) -> State:

    """Decide whether a policy document is needed"""
    chain = RunnablePassthrough.assign(input=(lambda x: x["messages"][-1])) | needs_policy_prompt | llm
    message = chain.invoke(state)
    state = interpret_policy_decision(message, state)
    return state


def initiate(state: State) -> State:
    """Initialise the state – ensure intermediate_outputs is set up as a dict"""

    state["intermediate_outputs"] = {}
    return state


# ------------


def create_retrieval_graph() -> CompiledStateGraph:  # doing it as a function to avoid circular imports
    """Compile and return a graph to assist with retrieval"""

    builder = StateGraph(State)

    #   builder.add_node("decide_if_person_page", decide_if_person_page)
    #  builder.add_node("decide_if_need_time_constraint", decide_if_need_time_constraint)
    # builder.add_edge(START, "decide_if_person_page")
    # builder.add_edge("decide_if_person_page", "decide_if_need_time_constraint")
    # builder.add_edge("decide_if_need_time_constraint", END)
    builder.add_node("initiate", initiate)
    builder.add_node("decide_whether_needs_policy", decide_whether_needs_policy)
    builder.add_edge(START, "initiate")
    builder.add_edge("initiate", "decide_whether_needs_policy")
    builder.add_edge("decide_whether_needs_policy", END)

    return builder.compile()


# graph = RunnableSequence(decide_if_person_page, decide_if_need_time_constraint)   #for comparison


# ----------chat graph


currentness_comment_chain = (
    RunnableParallel(
        input=(lambda x: x["messages"][-2]),
        answer=(lambda x: x["messages"][-1]),
    )
    | currentness_comment_prompt
    | llm
)


def currentness_comment(state: State, writer: StreamWriter) -> State:
    """Test example commenting on whether context is current"""

    sub_state = deepcopy(state)
    # the chain will interpret the references of the last CustomAIMessage as context
    # overwrite the references with only those which were actually cited in the response
    # it needs to be a copy as otherwise the references will be changed permanently
    #  and won't be listed properly at the end of the answer
    sub_state["messages"][-1].references = sub_state["messages"][-1].cited_references

    response = currentness_comment_chain.invoke(sub_state)
    if response.content:
        state["messages"][-1].content += "<br><br>" + response.content

    return state


def call_chain_func(**kwargs) -> State:
    """Return a function that calls the chain determined by kwargs"""

    rag_chain = importlib.import_module("llm.chain").get_graph_or_rag_chain(**kwargs)  # avoiding circular import

    def call_chain(
        state: State,
    ) -> State:  # function defined here to avoid circular import

        response = rag_chain.invoke(state)
        state["messages"].append(CustomAIMessage(response))

        return state

    return call_chain


def call_default_chain(state: State) -> State:
    """Call the default chain (i.e. the one returned by get_graph_or_rag_chain with no kwargs)"""

    return call_chain_func()(state)


def create_chat_graph(
    return_stream_nodes: bool = False, **kwargs
) -> CompiledStateGraph:  # doing it as a function to avoid circular imports
    """Compile and return a graph to assist with chat"""

    call_default_chain = call_chain_func(**kwargs)

    builder = StateGraph(State)

    builder.add_node("initiate", initiate)
    builder.add_node("call_default_chain", call_default_chain)
    builder.add_node("currentness_comment", currentness_comment)
    builder.add_edge(START, "initiate")
    builder.add_edge("initiate", "call_default_chain")
    builder.add_edge("call_default_chain", "currentness_comment")
    builder.add_edge("currentness_comment", END)

    stream_nodes = ["currentness_comment"]  # list of nodes whose outputs are to be streamed IN ORDER

    graph = builder.compile()

    if return_stream_nodes:
        return graph, stream_nodes
    else:
        return graph


# ----------combined graph


def choose_main_prompt(state: State) -> State:
    """Choose the main prompt based on whether retrieval has been restricted to policy documents"""

    if state["intermediate_outputs"].get("policy_file_ids"):
        main_prompt = qa_verbatim_prompt
    else:
        main_prompt = qa_prompt

    state["intermediate_outputs"]["main_prompt"] = main_prompt

    return state


def create_combined_graph(
    return_stream_nodes: bool = False, **kwargs
) -> CompiledStateGraph:  # doing it as a function to avoid circular imports
    """Compile and return a graph to assist with both retrieval and chat"""

    def call_chain(
        state: State,
    ) -> State:  # function defined here to avoid circular import

        prompt = state["intermediate_outputs"].get("main_prompt")
        rag_chain = importlib.import_module("llm.chain").get_graph_or_rag_chain(prompt=prompt, **kwargs)
        response = rag_chain.invoke(state)
        state["messages"].append(CustomAIMessage(response))

        return state

    builder = StateGraph(State)

    builder.add_node("initiate", initiate)
    builder.add_node("decide_whether_needs_policy", decide_whether_needs_policy)
    builder.add_node("choose_main_prompt", choose_main_prompt)
    builder.add_node("call_chain", call_chain)
    builder.add_edge(START, "initiate")
    builder.add_edge("initiate", "decide_whether_needs_policy")
    builder.add_edge("decide_whether_needs_policy", "choose_main_prompt")
    builder.add_edge("choose_main_prompt", "call_chain")
    builder.add_edge("call_chain", END)

    stream_nodes = ["call_chain"]  # list of nodes whose outputs are to be streamed IN ORDER

    graph = builder.compile()

    if return_stream_nodes:
        return graph, stream_nodes
    else:
        return graph


if __name__ == "__main__":

    graph = create_retrieval_graph()
    #  input = "What work has Nesta done on heat pumps?"
    # input = "Who has data science skills at Nesta?"
    # input = 'Are you a lemon?'
    #  input = "List all the reports published last year"
    input = "List all the reports published recently"
    # messages =
    res = asyncio.run(graph.ainvoke({"input": input, "filter_condition": "", "merge": True}))
    logger.info(res)

    # for m in messages['messages']:
    #   m.pretty_print()
