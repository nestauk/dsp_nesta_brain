from __future__ import annotations

import ast
import importlib

from os import system
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Literal
from typing import Type

import lgraph.research_agent.prompt as pt
import streamlit as st

from dsp_nesta_brain import logger
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from lgraph.graph import State
from llm.chain import create_retrieval_chain
from llm.llm import default_llm as llm
from llm.message import CustomAIMessage

# from llm.prompt import qa_prompt
from retrieval.retrieve import CustomRetriever


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


MAX_REVISIONS = 3

pause = input

# Credits:
# the code in this file and lgraph/research_agent/prompt.py is inspired by and in places adapted
# from the following two examples of research agents:
# GPT Researcher https://github.com/assafelovic/gpt-researcher/tree/master
# by Assaf Elovic and collaborators
# https://medium.com/towards-data-science/building-a-research-agent-that-can-write-to-google-docs-part-1-4b49ea05a292
# by Robert Martin-Short


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
    context: List[Dict]
    sidebar_options: Dict
    edited: bool


def call_model(state: AgentState, prompt: PromptTemplate) -> AgentState:
    """Call the LLM to generate a response to a prompt."""

    chain = RunnablePassthrough.assign(request=(lambda x: x["messages"][-1])) | prompt | llm
    result = chain.invoke(state)
    return result


def call_retrieval_chain(state: AgentState) -> AgentState:
    """Call a retrieval chain to generate a response to a prompt."""

    prompt = PromptTemplate(
        template=pt.get_write_prompt(state).template + "\nContext:\n{context}", input_variables=["request", "context"]
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

    use_retrieval_sidebar_option = (
        importlib.import_module("app_research").WIDGET_SPEC["knowledge_source"].get("use_retrieval_option")
    )
    use_retrieval = state.get("sidebar_options", {}).get("knowledge_source") == use_retrieval_sidebar_option

    st.toast(f'Writing draft no. {state.get("revision_number",0)+1}', icon="💡")

    if use_retrieval:
        result = call_retrieval_chain(state)
        state["draft"] = result["answer"]
        state["context"] = result["context"]
    else:
        result = call_model(state, pt.get_write_prompt(state))
        state["draft"] = result.content
        state["context"] = []

    return state


def review(state: AgentState) -> AgentState:
    """Review the draft of the document."""

    st.toast(f'Reviewing draft no. {state.get("revision_number",0)+1}', icon="💡")

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
    st.toast(f"Revising draft no. {state.get('revision_number',0)}", icon="💡")

    result = call_model(state, pt.revision_prompt)
    result = ast.literal_eval(result.content)
    state["draft"] = result.get("draft")
    state["revision_notes"] = result.get("revision_notes")
    return state


def terminate(state: AgentState) -> AgentState:
    """Reject the draft of the document."""

    if state.get("edited"):
        output_format = "Document after editing:\n\n{draft}"
    elif state.get("finalized_state"):
        output_format = "Document after {revision_number} revision(s):\n\n{draft}"
    else:
        output_format = "Document after {revision_number} revision(s) (NB: revision process was not fully completed):\n\n{draft}"  # noqa

    custom_ai_message_input_dict = state
    custom_ai_message_input_dict["answer"] = output_format.format(**state)
    state["messages"].append(CustomAIMessage(custom_ai_message_input_dict))

    return state


# copied from https://medium.com/towards-data-science/building-a-research-agent-that-can-write-to-google-docs-part-1-4b49ea05a292
def should_continue(state: AgentState) -> Literal["terminate", "revise"]:
    """
    Determine whether the research process should continue based on the current state.

    Args:
        state (AgentState): The current state of the research agent.

    Returns:
        str: The next node to transition to
    """
    # always send to review if editor hasn't made comments yet

    if state.get("finalized_state"):
        return "terminate"
    elif state.get("revision_number", 0) >= MAX_REVISIONS:
        logger.info("The number of revisions has reached the maximum ... terminating")
        return "terminate"
    else:
        return "revise"


def create_research_agent(editable: bool = False) -> CompiledStateGraph:
    """Create the research agent."""

    agent = StateGraph(AgentState)

    # agent.add_node("initial_plan", nodes.plan_node)
    agent.add_node("write", write)
    agent.add_node("review", review)
    # if False:
    agent.add_node("terminate", terminate)
    agent.add_node("revise", revise)

    # Edges
    agent.add_edge(START, "write")
    agent.add_edge("write", "review")
    agent.add_edge("revise", "review")
    agent.add_edge("terminate", END)
    agent.add_conditional_edges("review", should_continue)

    # stream_nodes = ["call_model"]  # list of nodes whose outputs are to be streamed IN ORDER

    if editable:
        memory = MemorySaver()
        return agent.compile(interrupt_before=["terminate"], checkpointer=memory)
    else:
        return agent.compile()


def create_graph(**kwargs) -> CompiledStateGraph:
    """Create the research agent graph."""

    OfficeTemplate: Type = importlib.import_module("lgraph.drive_doc.office_template").OfficeTemplate

    builder = StateGraph(AgentState)

    subgraph = create_research_agent(**kwargs)
    nodes = {"apply_template": subgraph}  # research agent subgraph will be used as a node

    agent = OfficeTemplate.sub_graph(builder=builder, **nodes)

    return agent


if __name__ == "__main__":

    toggle = True

    config = {}
    agent = create_graph()

    if toggle:

        input = "Write a project proposal for a new project assessing the recent uptake of heat pumps in the UK."

        input = {
            "messages": [HumanMessage(content=input)],
            "filter_condition": None,
            "limit": 10,
            "use_hybrid_search": True,
        }

        state = AgentState(input)

        config = {"configurable": {"thread_id": "1"}}

        agent.invoke(state, config=config)

    else:

        agent.get_graph().draw_mermaid_png(output_file_path="lgraph/research_agent/mermaid.png")
        system("open lgraph/research_agent/mermaid.png")  # nosec
