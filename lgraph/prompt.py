from datetime import datetime

from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from pydantic import Field


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

date_constraint_prompt = PromptTemplate(template=date_constraint_prompt_template, input_variables=["input", "answer"])


currentness_comment_prompt_template_1 = f"""
    You are a helpful RAG system and an expert on the internal administration, personnel and projects of the innovation agency Nesta. The year is currently {datetime.now().year}.

    You were asked to search for documents relevant to the question below and provided an answer and appropriate context.

    Look at the context metadata and comment on whether the answer to your question was likely to be currently correct.

    Question:
    {{input}}

    Answer:
    {{answer}}
    """  # noqa

currentness_comment_prompt_template_2 = f"""
    You are a helpful RAG system and an expert on the internal administration, personnel and projects of the innovation agency Nesta. The year is currently {datetime.now().year}.

    You were asked to search for documents relevant to the question below and you provided an answer and appropriate context. However, the answer might be based on out-of-date context.

    Look at the context metadata and rephrase the answer based on assessment of whether the context is providing up-to-date information. Express doubt about the answer if it isn't. Limit your answer to 100 words.

    Question:
    {{input}}

    Answer:
    {{answer}}
    """  # noqa


currentness_comment_prompt_template_3 = f"""
    You are a helpful RAG system and an expert on the internal administration, personnel and projects of the innovation agency Nesta. Today's date is {datetime.now().date()}.

    You were asked to search for documents relevant to the question below and you provided an answer and appropriate context. However, the answer might be based on out-of-date context.

    Look at the context metadata and assess whether the context is providing up-to-date information. If the context is over a year old, mention that the information on which you are basing your answer could be out of date. Limit your answer to 100 words.

    Question:
    {{input}}

    Answer:
    {{answer}}
    """  # noqa


currentness_comment_prompt_template_4 = f"""
    You are a helpful RAG system and an expert on the internal administration, personnel and projects of the innovation agency Nesta. Today's date is {datetime.now().date()}.

    You were asked to search for documents relevant to the question below and you provided an answer and appropriate context. However, the answer might be based on out-of-date context.

    Look at the context metadata. If the context is over a year old, mention that the information on which you are basing your answer could be out of date. Limit your answer to 50 words. If all of the context is less than a year old, return an empty string.

    Question:
    {{input}}

    Answer:
    {{answer}}
    """  # noqa

currentness_comment_prompt = PromptTemplate(
    template=currentness_comment_prompt_template_4, input_variables=["input", "answer"]
)


class OfficeTemplate(BaseModel):
    """A class for describing templates for proposals, project updates, etc."""

    UID: str = (
        Field(
            ...,
            description="A unique identifier",
        ),
    )
    title: str = (
        Field(
            ...,
            description="The template title",
        ),
    )
    purpose: str = (
        Field(
            ...,
            description="What the template is for",
        ),
    )
    location: str

    def __repr__(self) -> str:
        """Self-explanatory"""
        format_ = "\n\tUID: {UID}\n\ttitle: {title}\n\tpurpose: {purpose}\n"
        return format_.format(**{k: getattr(self, k) for k in self.__class__.dict(self) if k in format_})


office_templates = [
    OfficeTemplate(
        UID="PJPROP",
        title="Project Proposal Template",
        purpose="Help staff write proposals at the Opportunity and Scoping phases",
        location="https://docs.google.com/document/d/1cOv2vXcIPWQmRPeVDB-JMz8rS1CUIkyUeCFR-_7HyqY",
    ),
    OfficeTemplate(
        UID="PJUPDT",
        title="Project Updates Guidance",
        purpose="Help staff write webpages informing the public about project updates",
        location="https://docs.google.com/document/d/1cOv2vXcIPWQmRPeVDB-JMz8rS1CUIkyUeCFR-_7HyqY",
    ),
]


needs_template_template = f"""
    You are a helpful assistant and an expert on the internal administration, personnel and projects of the innovation agency Nesta.

    You have the following office document templates to help staff do their work: {office_templates}

    Look at the question below and decide whether the staff member who asked it needs one of the document templates.

    If so, respond just with the template UID. Otherwise, respond "NULL".

    Question:
    {{input}}
    """  # noqa


needs_template_prompt = PromptTemplate(template=needs_template_template, input_variables=["input"])
