from __future__ import annotations

import ast

from os import system

import lgraph.research_agent.prompt as pt

from dsp_nesta_brain import logger
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from lgraph.graph import State
from llm.chain import create_retrieval_chain
from llm.llm import default_llm as llm

# from llm.prompt import qa_prompt
from retrieval.retrieve import CustomRetriever


# from typing import TYPE_CHECKING


MAX_REVISIONS = 2


# inspiration:
# https://medium.com/towards-data-science/building-a-research-agent-that-can-write-to-google-docs-part-1-4b49ea05a292


class AgentState(State):
    """
    A dictionary representing the state of the research agent.

    Attributes:
        draft (str): The current draft of the research report.
        critique (str): The critique received for the draft.
        revision_notes (str): Most recent changes made to the draft
        revision_number (int): The current revision number of the draft.
        finalized_state (bool): Indicates whether the report is finalized.
    """

    draft: str
    critique: str
    revision_notes: str
    revision_number: int
    finalized_state: bool = False


def call_model(state: AgentState, prompt: PromptTemplate) -> AgentState:
    """Call the LLM to generate a response to a prompt."""

    chain = RunnablePassthrough.assign(request=(lambda x: x["messages"][-1])) | prompt | llm
    result = chain.invoke(state)
    return result


def call_retrieval_chain(state: AgentState) -> AgentState:
    """Call a retrieval chain to generate a response to a prompt."""

    prompt = PromptTemplate(
        template=pt.write_prompt.template + "\nContext:\n{context}", input_variables=["request", "context"]
    )
    chat_qa_chain = RunnablePassthrough.assign(request=(lambda x: x["messages"][-1])) | create_stuff_documents_chain(
        llm, prompt
    )
    retrieval_chain = create_retrieval_chain(CustomRetriever(), chat_qa_chain)
    result = retrieval_chain.invoke(state)
    return result


def write(state: AgentState) -> AgentState:
    """
    Write the initial draft of the document.

    Args:
        state (AgentState): The current state of the research agent.

    Returns:
        AgentState: The updated state of the research agent.
    """

    use_retrieval = True

    if use_retrieval:
        result = call_retrieval_chain(state)
        state["draft"] = result
    else:
        result = call_model(state, pt.write_prompt)
        state["draft"] = result.content

    return state


def review(state: AgentState) -> AgentState:
    """Review the draft of the document."""

    result = call_model(state, pt.get_review_prompt(state))
    if result.content == "NULL":
        state["finalized_state"] = True
    else:
        state["critique"] = result.content
        logger.info(
            f"The critique of the draft at revision number {state.get('revision_number',0)} was as follows:\n--------------------------------{state['critique']}\n--------------------------------"  # noqa
        )
    return state


def revise(state: AgentState) -> AgentState:
    """Revise the draft of the document."""

    state["revision_number"] = state.get("revision_number", 0) + 1
    logger.info(f"Undertaking revision number {state['revision_number']}")
    result = call_model(state, pt.revision_prompt)
    result = ast.literal_eval(result.content)
    state["draft"] = result.get("draft")
    state["revision_notes"] = result.get("revision_notes")
    return state


def terminate(state: AgentState) -> AgentState:
    """Reject the draft of the document."""
    logger.info("Draft terminated")
    return state


def accept(state: AgentState) -> AgentState:
    """Accept the draft of the document."""
    logger.info("Draft accepted")
    return state


# copied from https://medium.com/towards-data-science/building-a-research-agent-that-can-write-to-google-docs-part-1-4b49ea05a292
def should_continue(state: AgentState) -> str:
    """
    Determine whether the research process should continue based on the current state.

    Args:
        state (AgentState): The current state of the research agent.

    Returns:
        str: The next node to transition to ("to_review", "accepted", or "terminated").
    """
    # always send to review if editor hasn't made comments yet

    if state.get("finalized_state"):
        return "accept"
    elif state.get("revision_number", 0) >= MAX_REVISIONS:
        logger.info("The number of revisions has reached the maximum ... terminating")
        return "terminate"
    else:
        return "revise"


agent = StateGraph(AgentState)

# agent.add_node("initial_plan", nodes.plan_node)
agent.add_node("write", write)
agent.add_node("review", review)
# if False:
agent.add_node("terminate", terminate)
agent.add_node("revise", revise)
agent.add_node("accept", accept)


# Edges
agent.add_edge(START, "write")
agent.add_edge("write", "review")
agent.add_edge("revise", "review")
agent.add_edge("accept", END)
agent.add_edge("terminate", END)
agent.add_conditional_edges("review", should_continue)


# stream_nodes = ["call_model"]  # list of nodes whose outputs are to be streamed IN ORDER

graph = agent.compile()


if __name__ == "__main__":

    toggle = True

    if toggle:

        input = "Write a project proposal for a new project assessing the recent uptake of heat pumps in the UK."

        input = {
            "messages": [HumanMessage(content=input)],
            "filter_condition": None,
            "limit": 10,
            "use_hybrid_search": True,
        }

        state = AgentState(input)
        state = graph.invoke(state)

        for k, v in state.items():
            print("\n", f"{k}: {v}")  # noqa

    else:

        graph.get_graph().draw_mermaid_png(output_file_path="lgraph/research_agent/mermaid.png")
        system("open lgraph/research_agent/mermaid.png")  # nosec
