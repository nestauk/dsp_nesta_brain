from typing import TYPE_CHECKING

from langchain_core.prompts import PromptTemplate
from lgraph.graph import State


if TYPE_CHECKING:
    pass


# Credits:
# the revision and review prompts in this file have been copied (with only slight changes) from:
# GPT Researcher
# Citation: Elovic, A. (2023). gpt-researcher (Version 0.5.4) [Computer software]. https://github.com/assafelovic/gpt-researcher
# See:
# https://github.com/assafelovic/gpt-researcher/blob/master/multi_agents/agents/reviser.py
# https://github.com/assafelovic/gpt-researcher/blob/master/multi_agents/agents/reviewer.py
# for the original code


def get_write_prompt(state: State) -> PromptTemplate:
    """Return a prompt for writing the first draft based on an office template"""

    google_doc_template = state["intermediate_outputs"]["template"]

    write_template = f"""
        You are a helpful assistant and an expert on the internal administration, personnel and projects of the innovation agency Nesta.

        Your job is to apply the given template to structure the output of the request.

        REQUEST:
        {{request}}


        TEMPLATE:
        {google_doc_template.text}
        Finally, the draft should have a title which reflects its content.

    """  # noqa

    return PromptTemplate(template=write_template, input_variables=["request"])


revision_format = """
{{
  "draft": The revised draft that you are submitting for review
  "revision_notes": Your message to the reviewer about the changes you made to the draft based on their feedback
}}
"""


revision_template = f"""
You have been tasked by your reviewer with revising the following draft.
Read the reviewer's notes, then write a new draft and make sure to address all of the points they raised.
Please keep all other aspects of the draft the same.
You MUST return nothing but a JSON in the following format:
{revision_format}
\nDraft:\n{{draft}}"\n\n"Reviewer's notes:\n{{critique}}\n\n
"""

revision_prompt = PromptTemplate(template=revision_template, input_variables=["draft", "critique"])


reviser_addendum = """\nThe reviser has already revised the draft based on your previous review notes with the following feedback:
{revision_notes}\n
Please provide additional feedback ONLY if critical since the reviser has already made changes based on your previous feedback.
If you think the article is sufficient or that non critical revisions are required, please aim to return 'NULL'.
"""


def get_review_prompt(state: State) -> PromptTemplate:
    """Return a prompt for reviewing a draft based on an office template"""

    google_doc_template = state["intermediate_outputs"]["template"]

    prompt_template_1 = f"""
    You have been tasked with reviewing the draft which was written based on a specific template.
    Send it for revision, along with your notes to guide the revision.
    DO NOT modify the template structure, for example, by suggesting new sections like an Executive Summary or Conclusions if they are not in the template.
    {reviser_addendum if state.get('revision_notes') else ""}

    Template: {google_doc_template.text}\nDraft: {{draft}}\n
    """  # noqa

    prompt_template_2 = f"""
    You have been tasked with reviewing the draft which was written based on a specific template.
    Please accept the draft if it is good enough to publish, or send it for revision, along with your notes to guide the revision.
    If not all of the template criteria are met, you should send appropriate revision notes.
    If the draft meets all the guidelines, please return 'NULL'.
    {reviser_addendum if state.get('revision_notes') else ""}

    Template: {google_doc_template.text}\nDraft: {{draft}}\n
    """  # noqa

    prompt_template_3 = f"""
    You have been tasked with reviewing the draft which was written based on a specific template.
    Please accept the draft if it is good enough to publish, or send it for revision, along with your notes to guide the revision.
    If the draft does not fit the provided template structure, you should include this in your revision notes.
    DO NOT modify the template structure, for example, by suggesting new sections like an Executive Summary or Conclusions if they are not in the template.
    If the draft meets all the guidelines, please return 'NULL'.
    {reviser_addendum if state.get('revision_notes') else ""}

    Template: {google_doc_template.text}\nDraft: {{draft}}\n
    """  # noqa

    if state.get("revision_number", 0) == 0:
        prompt_template = prompt_template_1  # do at least one revision
    else:
        prompt_template = prompt_template_3

    input_variables = ["draft"]
    if state.get("revision_notes"):
        input_variables.append("revision_notes")

    return PromptTemplate(template=prompt_template, input_variables=input_variables)
