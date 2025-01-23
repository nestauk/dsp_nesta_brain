from datetime import datetime

from config import DEFAULT_START_YEAR
from config import PROJECT


CURRENT_YEAR = datetime.now().year

if PROJECT == "NESTA_BRAIN":

    intro = """
            <h2>🧠 Nesta Brain</h2><br/>
            This is a prototype AI chatbot designed to help you explore Nesta's knowledge.
            It searches thousands of webpages and reports to find the most relevant content
            in response to your questions.
            <br/><br/>
            We hope this can support knowledge management by making it easier to locate information
            about past projects,
            and generate new outputs.
            <br/><br/>
            This is an early version and we welcome your feedback
            very much - please use the
            emojis below to highlight specific responses, and <a href='https://forms.gle/TwXqUMHNTaPbYC4e7'>leave
            us general feedback using this form</a>.
            You can also contact directly Karlis Kanders or Helen Jackson (Data Science Practice / Discovery Hub)
            on <a href="https://nesta.slack.com/archives/C05BCUZNATG">#proj-nesta-brain</a>.
            <br/><br/>
            The chatbot currently accesses information from <strong>Nesta's public website (up to October 2024)</strong>
            and does <strong>not</strong> include internal documents or systems like Nesta:Net, Slack, or GitHub.
            <br/><br/>
            Use the sidebar to customize the chatbot's search parameters, such as date range or mission team.
            Note that user queries and responses are saved for chatbot's performance evaluation and improvement.
            """

    WIDGET_SPEC = {
        "from_year": {
            "default": DEFAULT_START_YEAR,
            "filter_condition_format": "source.date_pub >= to_timestamp('{current_value}-01-01')",
        },
        "to_year": {
            "default": CURRENT_YEAR,
            "filter_condition_format": "source.date_pub <= to_timestamp('{current_value}-12-31')",
        },
        "include_people": {"default": "Yes", "filter_condition_format": "source.contentType != 'person page'"},
        "mission": {"default": None, "filter_condition_format": "array_contains(source.missions,'{current_value}')"},
    }

elif PROJECT == "POLICY_ATLAS":

    intro = """
            <h2>Policy Atlas</h2><br/>
            This is a prototype AI chatbot designed to help you find information relating to activities in the
            International Aid Transparency Initiative Datastore.
            """

    WIDGET_SPEC = {
        "from_year": {
            "default": DEFAULT_START_YEAR,
            "filter_condition_format": "min_year <= {current_value} and max_year >= {current_value}",
        },
        "to_year": {
            "default": CURRENT_YEAR,
            "filter_condition_format": "min_year <= {current_value} and max_year >= {current_value}",
        },
    }
