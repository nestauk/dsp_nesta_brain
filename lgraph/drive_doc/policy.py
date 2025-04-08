from __future__ import annotations

import importlib

from typing import TYPE_CHECKING
from typing import Callable

from google_api.drive_doc.policy import Policy
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from lgraph.drive_doc.base import BaseDriveDoc
from llm.prompt import qa_prompt as default_prompt
from llm.prompt import qa_verbatim_prompt


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from lgraph.graph import State


class Policy(Policy, BaseDriveDoc):
    """A class for describing Nesta policy documents."""

    @classmethod
    def decide_whether_needs_document(cls, state: State) -> State:
        """Decide whether a policy document is needed to answer a query base on a state and amend the state accordingly if so"""

        state = super(cls, Policy).decide_whether_needs_document(state)
        file_ids = state["intermediate_outputs"].get("file_ids")

        if file_ids:

            filter_condition = "(" + " or ".join([f'source.location LIKE "%{file_id}"' for file_id in file_ids]) + ")"
            state["use_hybrid_search"] = False
            # filter_condition = f'(source.drive_type == "policy" or source.location LIKE "%{file_id}")'
            state = importlib.import_module("lgraph.graph").append_filter_condition(
                state, filter_condition
            )  # avoiding circular import

        return state

    @classmethod
    def prompt_template(cls) -> str:
        """Return a string for a prompt template to use in a LangGraph node relating to the document"""

        return """
            You are a helpful assistant and an expert in the internal administration and organisational policies of the innovation agency Nesta.

            Your role is to help staff with their queries about organisational policies. Topics include annual leave, expenses, safeguarding, and more.

            Look at the following list of policy documents available to help you answer queries:

            List:
            {list}

            Look at the query below and decide whether one or more the policies in the list is needed to answer it.

            If so, select the appropriate policies. As your response, give only the UIDs of the policy you have selected separated by commas.

            Otherwise, respond "NULL".

            Query:
            {{input}}
            """  # noqa

    @classmethod
    def sub_graph(cls, call_model: Callable) -> CompiledStateGraph:
        """Return the subgraph for the OfficeTemplate class"""

        State = importlib.import_module("lgraph.graph").State

        builder = StateGraph(State)

        builder.add_node("decide_whether_needs_policy", cls.decide_whether_needs_document)
        builder.add_node("choose_main_prompt", choose_main_prompt)
        builder.add_node("call_model", call_model)

        builder.add_edge(START, "decide_whether_needs_policy")
        builder.add_edge("decide_whether_needs_policy", "choose_main_prompt")
        builder.add_edge("choose_main_prompt", "call_model")
        builder.add_edge("call_model", END)

        graph = builder.compile()
        return graph


def choose_main_prompt(state: State) -> State:
    """Choose the main prompt based on whether retrieval has been restricted to policy documents"""

    if state["intermediate_outputs"].get("file_ids"):
        main_prompt = qa_verbatim_prompt
    else:
        main_prompt = default_prompt

    state["intermediate_outputs"]["main_prompt"] = main_prompt

    return state
