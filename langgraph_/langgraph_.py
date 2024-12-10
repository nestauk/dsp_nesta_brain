from dotenv import load_dotenv
from dsp_nesta_brain import logger
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph_.prompt import personnel_prompt
from retrieval.retrieve import RetrieverInput


State = RetrieverInput


# -------NODES
def decide_if_person_page(state: State) -> State:
    """Decide if the retrieval should be limited to person pages only; add appropriate filter_condition to the state if so"""
    chain = personnel_prompt | routing_llm
    response = chain.invoke(state["input"])
    if response.content == "YES":
        state["filter_condition"] = "source.contentType = 'person page'"
    else:
        state["filter_condition"] = ""
    #  print("within LangGraph: ", state)
    return state


# ------------


load_dotenv()

routing_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

builder = StateGraph(State)

builder.add_node("decide_if_person_page", decide_if_person_page)
builder.add_edge(START, "decide_if_person_page")
builder.add_edge("decide_if_person_page", END)


graph = builder.compile()


if __name__ == "__main__":

    input = "Who has data science skills at Nesta?"
    # messages =
    res = graph.invoke({"input": input, "filter_condition": "", "merge": True})
    logger.info(res)

    # for m in messages['messages']:
    #   m.pretty_print()
