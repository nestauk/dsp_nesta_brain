from __future__ import annotations

from datetime import datetime
from typing import List

import streamlit as st

from config import EARLIEST_YEAR
from front_end.project_spec import WIDGET_SPEC


CURRENT_YEAR = datetime.now().year


def sidebar() -> List:
    """Return a list of sidebar widgets according to a project-specific WIDGET_SPEC"""

    sidebar_elements = []

    for key, spec in WIDGET_SPEC.items():

        if key == "from_year":

            from_year = st.number_input(
                label="From year",
                min_value=EARLIEST_YEAR,
                max_value=CURRENT_YEAR,
                key=key,
                value=spec["default"],
            )
            sidebar_elements.append(from_year)

        if key == "to_year":

            sidebar_elements.append(
                st.number_input(
                    label="To year",
                    min_value=from_year,
                    max_value=CURRENT_YEAR,
                    key=key,
                    value=spec["default"],
                )
            )

        if key == "include_people":
            sidebar_elements.append(
                st.radio(
                    "Include people pages",
                    spec["options"],
                    key=key,
                    index=spec["options"].index(spec["default"]),
                )
            )

        if key == "mission":
            sidebar_elements.append(
                st.radio(
                    "Mission-specific content",
                    spec["options"],
                    key=key,
                    index=spec["options"].index(spec["default"]),
                )
            )

    return sidebar_elements
