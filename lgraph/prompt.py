from datetime import datetime

from langchain.prompts import PromptTemplate


personnel_prompt_template = """
    You are a helpful assistant and an expert on the internal administration, personnel and projects of the innovation agency Nesta.

    Look at the question below and decide whether it is about personnel and best answered by looking at staff biographies and CVs. Answer "YES" or "NO".

    Question:
    {input}
    """  # noqa

personnel_prompt = PromptTemplate(template=personnel_prompt_template, input_variables=["input"])


year_constraint_prompt_template = f"""
    You are a helpful assistant and an expert on the internal administration, personnel and projects of the innovation agency Nesta. The year is currently {datetime.now().year}.

    You are being asked to search for documents relevant to the question below. Look at the question and decide whether it should limit your search only to documents from a particular year range.

    Be strict and only give a year range if the question actually mentions some aspect of time.

    If so, give a start_year and end_year for the year range.

    If not, then the start_year = None and end_year = None.

    Question:
    {{input}}
    """  # noqa

year_constraint_prompt = PromptTemplate(template=year_constraint_prompt_template, input_variables=["input"])


date_constraint_prompt_template = f"""
    You are a helpful assistant and an expert on the internal administration, personnel and projects of the innovation agency Nesta. Todays' date is {datetime.strftime(datetime.now(),'%Y-%m-%d')}.

    You are being asked to search for documents relevant to the question below. Look at the question and decide whether you should limit your search only to documents from a particular date range.

    If so, give a start_date and end_date in the format 'YYYY-mm-dd'.

    If not, then start_date = None and end_date = None.

    Question:
    {{input}}
    """  # noqa

date_constraint_prompt = PromptTemplate(template=date_constraint_prompt_template, input_variables=["input"])
