import re
import time

import google_auth_oauthlib.flow
import streamlit as st

from front_end.auth.token_manager import AuthTokenManager
from googleapiclient.discovery import build


# credit: https://medium.com/@coding-otter
# https://medium.com/@coding-otter/google-oauth-in-streamlit-a-solution-that-finally-works-for-me-a212a79fec30


class Authenticator:
    """Stores authentication details and provides authentication functions"""

    def __init__(
        self,
        # allowed_users: list,
        secret_path: str,
        redirect_uri: str,
        token_key: str,
        cookie_name: str = "auth_jwt",
        token_duration_days: int = 1,
    ) -> None:

        st.session_state["connected"] = st.session_state.get("connected", False)
        #  self.allowed_users = allowed_users
        self.secret_path = secret_path
        self.redirect_uri = redirect_uri
        self.auth_token_manager = AuthTokenManager(
            cookie_name=cookie_name,
            token_key=token_key,
            token_duration_days=token_duration_days,
        )
        self.cookie_name = cookie_name

    def _initialize_flow(self) -> google_auth_oauthlib.flow.Flow:
        """"""  # noqa
        flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
            self.secret_path,
            scopes=[
                "openid",
                "https://www.googleapis.com/auth/userinfo.profile",
                "https://www.googleapis.com/auth/userinfo.email",
            ],
            redirect_uri=self.redirect_uri,
        )
        return flow

    def get_auth_url(self) -> str:
        """"""  # noqa
        flow = self._initialize_flow()
        auth_url, _ = flow.authorization_url(access_type="offline", include_granted_scopes="true")
        return auth_url

    def login(self) -> None:
        """Login button"""
        if not st.session_state["connected"]:
            auth_url = self.get_auth_url()
            st.write("Welcome to Nesta Brain")
            st.link_button("Login with Google", auth_url)

    def check_auth(self) -> None:
        """Check whether a user is authorised"""
        
        token = self.auth_token_manager.get_decoded_token()
        if token is not None:
            st.query_params.clear()
            st.session_state["connected"] = True
            st.session_state["user_info"] = {
                "email": token["email"],
                "oauth_id": token["oauth_id"],
            }
        #   st.rerun()  # this was in the original example but I found lead to odd, undesirable behaviour

        time.sleep(1)  # important for the token to be set correctly

        auth_code = st.query_params.get("code")
        st.query_params.clear()
        if auth_code:
            flow = self._initialize_flow()
            flow.fetch_token(code=auth_code)
            creds = flow.credentials

            oauth_service = build(serviceName="oauth2", version="v2", credentials=creds)
            user_info = oauth_service.userinfo().get().execute()
            oauth_id = user_info.get("id")
            email = user_info.get("email")

            if re.search(
                "@nesta.org.uk$", email
            ):  # allow any email address with a nesta.org.uk domain, rather than having a list of allowed users
                self.auth_token_manager.set_token(email, oauth_id)
                st.session_state["connected"] = True
                st.session_state["user_info"] = {
                    "oauth_id": oauth_id,
                    "email": email,
                }
            else:
                st.toast("Unauthorised: you must have a Nesta email address to use this app")
            # no rerun


# def logout(self) -> None:
#    """Logout"""
#    st.session_state["logout"] = True
#   st.session_state["user_info"] = None
#  st.session_state["connected"] = None
# self.auth_token_manager.delete_token()
# no rerun
