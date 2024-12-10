from langchain.prompts import PromptTemplate


personnel_prompt_template = """
    You are an experimental, helpful assistant and an expert on the internal administration, personnel and projects of the innovation agency Nesta.

    Look at the question below and decide whether it is about personnel and best answered by looking at staff biographies and CVs. Answer "YES" or "NO".

    Question: {input}
    """  # noqa

personnel_prompt = PromptTemplate(template=personnel_prompt_template, input_variables=["input"])
