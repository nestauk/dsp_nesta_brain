from collections import OrderedDict
from datetime import datetime

from config import DEFAULT_START_YEAR
from config import PROJECT


CURRENT_YEAR = datetime.now().year

if PROJECT == "NESTA_BRAIN":

    INTRO = """
            <h2>🧠 Nesta Brain</h2><br/>

            <span style="color:red">DELETE THIS: red text shows recent edits to the intro.
            Remove the red styling when happy with the edits.</span><br><br>

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
            <br/><br/>
            <span style="color: red">Like any AI chatbot, responses are not guaranteed to be 100% accurate and complete.
            Please always double-check the information provided and use your own judgement.
            If you need in-depth assistance on matters which could affect your personal welfare or career, please
            seek help from the relevant staff member.</span>
            """

    WIDGET_SPEC = OrderedDict(
        {
            "from_year": {
                "default": DEFAULT_START_YEAR,
                "filter_condition_format": "source.date_pub >= to_timestamp('{current_value}-01-01')",
            },
            "to_year": {
                "default": CURRENT_YEAR,
                "filter_condition_format": "source.date_pub <= to_timestamp('{current_value}-12-31')",
            },
            "include_people": {
                "default": "Yes",
                "filter_condition_format": "source.contentType != 'person page'",
                "options": ("Yes", "No"),
            },
            "mission": {
                "default": None,
                "filter_condition_format": "array_contains(source.missions,'{current_value}')",
                "options": ("A fairer start", "A healthy life", "A sustainable future", None),
            },
        }
    )

elif PROJECT == "POLICY_ATLAS":

    INTRO = """
            <h2>🌎 Policy Atlas</h2><br/>
            This is an early prototype AI chatbot designed to help you find and summarise information about
            international aid activities by the Foreign, Commonwealth and Development Office (FCDO).
            It includes both activities led by the FCDO and those where the FCDO is a partner.
            <br></br>
            You can ask questions such as: <em>"What are the activities related to education in Kenya?"</em> and
            <em>"How do we help improve the quality of water services in Africa?"</em>. Use the sidebar to constrain the
            search to a specific time period. For each user query, the chatbot processes the top 10 most relevant
            activities to the query. Therefore, for more precise answers, it is recommended to ask more specific
            questions. For more detail about each activity or to find statistics about aid spending, please visit
            <a href="https://devtracker.fcdo.gov.uk/">Development Tracker</a>.
            <br></br>
            Chatbot's data is sourced from the International Aid Transparency Initiative (IATI)
            <a href="https://datastore.iatistandard.org/">datastore</a> and includes about 38,000 active and closed
            activities covering the period of 2000–2025. Chatbot's data was last updated on 17 Jan 2025.
            A visualisation of the chatbot's data can also be viewed
            <a href="https://atlas.nomic.ai/data/kandersk/uk-aid-data/map">here</a>. The chatbot uses the OpenAI API for
            generating responses, and user queries are <em>not</em> used to train AI models.
            <br></br>
            Note that this prototype is developed by <a href="https://www.nesta.org.uk/">Nesta</a> and is <em>not</em>
            a product of the FCDO or the UK government.
            For questions and comments, please reach out to karlis.kanders@nesta.org.uk.
            <br></br>
            """

    WIDGET_SPEC = OrderedDict(
        {
            "from_year": {
                "default": DEFAULT_START_YEAR,
                "filter_condition_format": "min_year <= {current_value} and max_year >= {current_value}",
            },
            "to_year": {
                "default": CURRENT_YEAR,
                "filter_condition_format": "min_year <= {current_value} and max_year >= {current_value}",
            },
        }
    )
