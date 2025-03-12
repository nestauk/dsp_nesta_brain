from __future__ import annotations

from datetime import datetime
from typing import List

import streamlit as st

from config import EARLIEST_YEAR
from front_end.project_spec import WIDGET_SPEC as DEFAULT_WIDGET_SPEC


CURRENT_YEAR = datetime.now().year


def sidebar(*args) -> List:
    """Return a list of sidebar widgets according to a project-specific WIDGET_SPEC"""

    if args:
        WIDGET_SPEC = args[0]
    else:
        WIDGET_SPEC = DEFAULT_WIDGET_SPEC

    sidebar_elements = []

    for key, spec in WIDGET_SPEC.items():

        label = spec.get("label") or key.replace("_", " ").capitalize()

        if key == "from_year":

            from_year = st.number_input(
                label=label,
                min_value=EARLIEST_YEAR,
                max_value=CURRENT_YEAR,
                key=key,
                value=spec["default"],
            )
            sidebar_elements.append(from_year)

        if key == "to_year":

            sidebar_elements.append(
                st.number_input(
                    label=label,
                    min_value=from_year,
                    max_value=CURRENT_YEAR,
                    key=key,
                    value=spec["default"],
                )
            )

        elif spec.get("element_type") == "radio":
            sidebar_elements.append(
                st.radio(
                    label,
                    spec["options"],
                    key=key,
                    index=spec["options"].index(spec["default"]),
                )
            )

    return sidebar_elements
