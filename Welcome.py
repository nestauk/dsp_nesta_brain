from __future__ import annotations

import logging
import os

import streamlit as st

from config import DEBUG_MODE
from config import DEPLOY_MODE
from dotenv import load_dotenv
from front_end.auth.authenticate import Authenticator
from front_end.project_spec import WELCOME_INTRO


def setup() -> None:

    """Perform setup tasks for the app:
    - Load environment variables
    - Remove unwanted information from logging
    - Authentication
    """  # noqa

    load_dotenv()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # -------authentication credit------
    # credit: https://medium.com/@coding-otter
    # https://medium.com/@coding-otter/google-oauth-in-streamlit-a-solution-that-finally-works-for-me-a212a79fec30

    if DEPLOY_MODE:
        redirect_uri = "https://nesta-brain.dap-tools.uk/"
    else:
        redirect_uri = "http://localhost:8501"
    authenticator = Authenticator(  # allows any email address with a nesta.org.uk domain
        token_key=os.getenv("AUTH_TOKEN_KEY"),
        secret_path="client_secret.json",  # nosec
        redirect_uri=redirect_uri,
    )

    authenticator.check_auth()
    authenticator.login()


if __name__ == "__main__":

    setup()

    if st.session_state["connected"]:
        # st.set_page_config(layout="wide")

        st.sidebar.success("Select a page above.")

        st.markdown(
            """
        <style>
            p {
                margin-bottom: 0;
            }

            a{
                margin-top: 0;
            }

            .response {
                margin: 25px 0 0 0;
                background-color: light-grey;
            }

        </style>
        """,
            unsafe_allow_html=True,
        )

        if DEBUG_MODE:
            st.markdown(
                '<p style="color:red;font-size:125%"><b>WARNING: DEBUG MODE IS ON</b></p>', unsafe_allow_html=True
            )

        st.markdown(
            WELCOME_INTRO,
            unsafe_allow_html=True,
        )
