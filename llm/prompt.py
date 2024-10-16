import logging
import sys

from langchain.prompts import PromptTemplate


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


question_prompt_template = """You are a helpful assistant and an expert on the internal administration, personnel and projects of the innovation agency Nesta. Use the following pieces of context to extract facts which help answer the question. Keep your replies short and informative. If the context does not help you answer the question, reply 'NULL'.
    {context}
    Question: {question}
    """  # noqa

question_prompt = PromptTemplate(template=question_prompt_template, input_variables=["context", "question"])
