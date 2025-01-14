from __future__ import annotations

# import asyncio
import importlib

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import TypedDict

from dsp_nesta_brain import logger
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from lgraph.prompt import needs_template_prompt
from lgraph.prompt import office_templates
from llm.llm import default_llm as llm
from llm.message import CustomAIMessage
from retrieval.retrieve import RetrieverInput as InputState


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


LAST_CHAT_GRAPH_NODE_NAME = "currentness_comment"


class InternalState(TypedDict):
    """Defines state passed between nodes internally"""

    outputs: List[Any]


class OutputState(TypedDict):
    """Defines the state returned by the final node"""

    pass


# -----nodes


def determine_whether_needs_template(state: InputState) -> InternalState:
    """Determine whether an office template is needed"""

    def office_templates_as_dict() -> Dict:
        return {template.UID: template for template in office_templates}

    chain = RunnablePassthrough.assign(input=(lambda x: x["messages"][-1])) | needs_template_prompt | llm
    response = chain.invoke(state)

    template = office_templates_as_dict().get(response)
    state.outputs.append(template)

    return state


# -----conditional edges


def needs_template(state: InternalState) -> str:
    """Go the appropriate node, depending on whether an office template is needed"""

    if state.outputs[-1]:
        return "docgen"
    else:
        return "call_model"


def create_chat_graph(**kwargs) -> CompiledStateGraph:  # doing it as a function to avoid circular imports
    """Compile and return a graph to assist with chat"""

    rag_chain = importlib.import_module("llm.chain").history_aware_rag_chain(**kwargs)  # avoiding circular import

    def call_model(
        state: InternalState,
    ) -> OutputState:  # function defined here to avoid circular import

        response = rag_chain.invoke(state)
        state["messages"].append(CustomAIMessage(response))

        return state

    builder = StateGraph(InternalState)

    # builder.add_node("call_model", call_model)
    builder.add_node("determine_whether_needs_template", determine_whether_needs_template)
    builder.add_edge(START, "determine_whether_needs_template")
    builder.add_conditional_edge("determine_whether_needs_template", needs_template)
    builder.add_edge("doc_gen", END)
    builder.add_edge("call_model", END)

    return builder.compile()


if __name__ == "__main__":

    graph = create_chat_graph()
    input = "How do I write a project proposal?"
    input = "What does a lemon look like?"
    res = graph.invoke({"messages": [HumanMessage(content=input)], "filter_condition": "", "merge": True})
    logger.info(res)
