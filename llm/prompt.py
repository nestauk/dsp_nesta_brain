import logging
import sys

from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


basic_question_prompt_template = """You are a helpful assistant and an expert on the internal administration, personnel and projects of the innovation agency Nesta. Use the following pieces of context to extract facts which help answer the question. Keep your replies short and informative. If the context does not help you answer the question, reply 'NULL'.
    {context}
    Question: {question}
    """  # noqa

basic_question_prompt = PromptTemplate(
    template=basic_question_prompt_template, input_variables=["context", "question"]
)


qa_system_prompt = """You are a helpful assistant and an expert on the internal administration, personnel and projects of the innovation agency Nesta. Use the following pieces of context to extract facts which help answer the question. Keep your replies short and informative. If the context does not help you answer the question, reply 'NULL'.
    {context}
    """  # noqa

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


# credit: https://medium.com/@eric_vaillancourt/mastering-langchain-rag-integrating-chat-history-part-2-4c80eae11b43
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
