from __future__ import annotations

import asyncio
import importlib

from copy import deepcopy
from typing import TYPE_CHECKING

from config import DEFAULT_START_YEAR
from dsp_nesta_brain import logger
from langchain_core.runnables.base import RunnableParallel
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.types import StreamWriter
from lgraph.prompt import currentness_comment_prompt
from lgraph.prompt import personnel_prompt
from lgraph.prompt import year_constraint_prompt
from llm.llm import default_llm as llm
from llm.message import CustomAIMessage
from llm.tool import year_range
from retrieval.retrieve import RetrieverInput as State


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


DEFAULT_FROM_YEAR_FILTER_CONDITION = f"source.date_pub >= to_timestamp('{DEFAULT_START_YEAR}-01-01')"
LAST_CHAT_GRAPH_NODE_NAME = "currentness_comment"


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
def decide_if_person_page(state: State) -> State:
    """Decide if the retrieval should be limited to person pages only; add appropriate filter_condition to the state if so"""
    if (
        "source.contentType" not in state["filter_condition"]
    ):  # if the user has explicitly set a filter condition via the UI, use that one and ignore the node
        chain = personnel_prompt | llm
        response = chain.invoke(state["input"])
        if response.content == "YES":
            state = append_filter_condition(state, "source.contentType = 'person page'")
    #  print("within 'decide_if_person_page': ", state)
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


# ------------


def create_retrieval_graph() -> CompiledStateGraph:  # doing it as a function to avoid circular imports
    """Compile and return a graph to assist with retrieval"""

    builder = StateGraph(State)

    builder.add_node("decide_if_person_page", decide_if_person_page)
    builder.add_node("decide_if_need_time_constraint", decide_if_need_time_constraint)
    builder.add_edge(START, "decide_if_person_page")
    builder.add_edge("decide_if_person_page", "decide_if_need_time_constraint")
    builder.add_edge("decide_if_need_time_constraint", END)

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


def create_chat_graph(**kwargs) -> CompiledStateGraph:  # doing it as a function to avoid circular imports
    """Compile and return a graph to assist with chat"""

    rag_chain = importlib.import_module("llm.chain").history_aware_rag_chain(**kwargs)  # avoiding circular import

    def call_model(
        state: State,
    ) -> State:  # function defined here to avoid circular import

        response = rag_chain.invoke(state)
        state["messages"].append(CustomAIMessage(response))

        return state

    builder = StateGraph(State)

    builder.add_node("call_model", call_model)
    builder.add_node("currentness_comment", currentness_comment)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", "currentness_comment")
    builder.add_edge("currentness_comment", END)

    return builder.compile()


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
