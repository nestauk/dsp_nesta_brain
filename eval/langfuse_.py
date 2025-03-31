from __future__ import annotations

import os

from langfuse import Langfuse
from langfuse.callback import CallbackHandler


# Credit: https://langfuse.com/docs/sdk/python/low-level-sdk
def get_langfuse() -> Langfuse:
    """Return Langfuse constructor, but only set it once"""

    if not hasattr(get_langfuse, "langfuse"):
        get_langfuse.langfuse = Langfuse()  # debug=True)

    return get_langfuse.langfuse


# langfuse = Langfuse(secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
#       public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
#  host=os.getenv("LANGFUSE_HOST"))#,
#  user_id=os.getenv("LANGFUSE_USER_ID"))

# langfuse = Langfuse()

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    user_id=os.getenv("LANGFUSE_USER_ID"),
)
