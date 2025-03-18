from datetime import datetime
from datetime import timedelta

import extra_streamlit_components as stx
import jwt
import streamlit as st

from jwt import ExpiredSignatureError


# credit: https://medium.com/@coding-otter
# https://medium.com/@coding-otter/google-oauth-in-streamlit-a-solution-that-finally-works-for-me-a212a79fec30


class AuthTokenManager:
    """Authentication token manager"""

    def __init__(
        self,
        cookie_name: str,
        token_key: str,
        token_duration_days: int,
    ) -> None:
        self.cookie_manager = stx.CookieManager()
        self.cookie_name = cookie_name
        self.token_key = token_key
        self.token_duration_days = token_duration_days
        self.token = None

    def get_decoded_token(self) -> str:
        """"""  # noqa
        self.token = self.cookie_manager.get(self.cookie_name)
        if self.token is None:
            return None
        self.token = self._decode_token()
        return self.token

    def set_token(self, email: str, oauth_id: str) -> None:
        """"""  # noqa
        exp_date = (datetime.now() + timedelta(days=self.token_duration_days)).timestamp()
        token = self._encode_token(email, oauth_id, exp_date)
        self.cookie_manager.set(
            self.cookie_name,
            token,
            expires_at=datetime.fromtimestamp(exp_date),
        )

    def delete_token(self) -> None:
        """"""  # noqa
        try:
            self.cookie_manager.delete(self.cookie_name)
        except KeyError:
            pass

    def _decode_token(self) -> str:
        """"""  # noqa
        try:
            decoded = jwt.decode(self.token, self.token_key, algorithms=["HS256"])
            return decoded
        except ExpiredSignatureError:
            st.toast("token expired, please login")
            self.delete_token()

    def _encode_token(self, email: str, oauth_id: str, exp_date: float) -> str:
        """"""  # noqa
        encoded = jwt.encode(
            {"email": email, "oauth_id": oauth_id, "exp": exp_date},
            self.token_key,
            algorithm="HS256",
        )
        return encoded
