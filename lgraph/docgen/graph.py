from __future__ import annotations

# import asyncio
import importlib

from typing import TYPE_CHECKING
from typing import Any
from typing import List
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from lgraph.docgen.prompt import apply_template_prompt
from lgraph.docgen.tool import get_template
from lgraph.prompt import needs_template_prompt
from llm.llm import default_llm as llm
from llm.message import CustomAIMessage
from retrieval.retrieve import RetrieverInput as InputState


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


LAST_CHAT_GRAPH_NODE_NAME = "currentness_comment"


class OverallState(InputState):
    """Defines the state used internally and returned by the final node"""

    outputs: List[Any]


# -----nodes


def determine_whether_needs_template(state: InputState) -> OverallState:
    """Determine whether an office template is needed"""

    llm_with_tool = llm.bind_tools([get_template], tool_choice="get_template")
    chain = (
        RunnablePassthrough.assign(input=(lambda x: x["messages"][-1]))
        | needs_template_prompt
        | llm_with_tool
        | (lambda x: get_template.invoke(x.tool_calls[0]["args"]))
    )

    template = chain.invoke(state)

    state["outputs"] = [template]

    return state


def apply_template(state: OverallState) -> OverallState:
    """Apply a template retrieved from Drive to format the input request"""

    chain = (
        RunnableParallel(
            request=(lambda state: state["messages"][-1].content), template=(lambda state: state["outputs"][-1].text)
        )
        | apply_template_prompt
        | llm
    )

    message = chain.invoke(state)

    state["messages"].append(message)

    return state


# -----conditional edges


def needs_template(state: OverallState) -> Literal["apply_template", "call_model"]:
    """Go the appropriate node, depending on whether an office template is needed"""

    office_template = state["outputs"][-1]
    if office_template:
        return "apply_template"  # "fetch_template"
    else:
        return "call_model"


# -----graph


def create_chat_graph(**kwargs) -> CompiledStateGraph:  # doing it as a function to avoid circular imports
    """Compile and return a graph to assist with chat"""

    rag_chain = importlib.import_module("llm.chain").history_aware_rag_chain(**kwargs)  # avoiding circular import

    def call_model(
        state: OverallState,
    ) -> OverallState:  # function defined here to avoid circular import

        response = rag_chain.invoke(state)
        state["messages"].append(CustomAIMessage(response))

        return state

    builder = StateGraph(OverallState)

    builder.add_node("call_model", call_model)
    builder.add_node("determine_whether_needs_template", determine_whether_needs_template)
    # builder.add_node("fetch_template", fetch_template)
    builder.add_node("apply_template", apply_template)
    builder.add_edge(START, "determine_whether_needs_template")
    builder.add_conditional_edges("determine_whether_needs_template", needs_template)
    #    builder.add_edge("fetch_template", 'apply_template')
    builder.add_edge("apply_template", END)
    builder.add_edge("call_model", END)

    return builder.compile()


if __name__ == "__main__":

    graph = create_chat_graph()
    input = "Help me write a project proposal for the 'Making heat pumps better' project"
    # input = "What does a lemon look like?"
    input = {"messages": [HumanMessage(content=input)], "filter_condition": "", "merge": True}
    res = graph.invoke(input)

    for message in res["messages"]:
        message.pretty_print()

    if False:
        llm_with_tool = llm.bind_tools([get_template], tool_choice="get_template")
        chain = (
            RunnablePassthrough.assign(input=(lambda x: x["messages"][-1]))
            | needs_template_prompt
            | llm_with_tool
            | (lambda x: get_template.invoke(x.tool_calls[0]["args"]))
        )
        # output_parser = JsonOutputKeyToolsParser(key_name="get_template", first_tool_only=True)
        # answer = needs_template_prompt | llm_with_tool | output_parser

        res = chain.invoke(input)

        print(res)  # noqa
