# from langchain_core.prompts import MessagesPlaceholder
from google_api.office_template import office_templates
from langchain_core.prompts import PromptTemplate


needs_template_template = f"""
    You are a helpful assistant and an expert on the internal administration, personnel and projects of the innovation agency Nesta.

    Look at the following list of office document templates available to you to help staff write documents: {office_templates}

    Look at the request below and decide whether the one of the document templates in the list is needed to fulfil it.

    If so, select the appropriate template. As your response, give only the UID of the template you have selected.

    Otherwise, respond "NULL".

    Request:
    {{input}}
    """  # noqa

# print(needs_template_template,'\n\n')

needs_template_prompt = PromptTemplate(template=needs_template_template, input_variables=["input"])

apply_template_template = """
    You an expert at writing office documents at Nesta. Your job is to apply the given template to structure the output of the request.

    REQUEST:
    {input}


    TEMPLATE:
    {template}

"""  # noqa

apply_template_prompt = PromptTemplate(template=apply_template_template, input_variables=["request", "template"])


apply_template_appendix = """

    Finally, apply the given template to structure your answer. Give only the final document in Markdown as your answer.

    TEMPLATE:
    {template}
"""
