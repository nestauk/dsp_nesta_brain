# from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import PromptTemplate


apply_template_template = """
    You an expert at writing office documents at Nesta. Your job is to apply the given template to structure the output of the request.

    REQUEST:
    {request}


    TEMPLATE:
    {template}

"""  # noqa

apply_template_prompt = PromptTemplate(template=apply_template_template, input_variables=["request", "template"])
