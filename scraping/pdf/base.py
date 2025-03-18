from __future__ import annotations

from datetime import date
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from dsp_nesta_brain import logger
from utils import yesno


class BasePDF:
    """Represents a scraped PDF document"""

    location: str
    elements: List

    def __init__(self, location: str, **kwargs) -> None:
        self.location = location
        logger.info(f"\nReading PDF document {self.location} ...")

    def guess_metadata(
        self,
        title_guess: Optional[Union[str, List[str]]] = None,
        date_guess: Optional[Union[str, date]] = None,
        cautious: bool = False,
        force_date: bool = False,
        indent: Optional[str] = "",
    ) -> Dict:
        """Guess the title and check whether the title guess and date guess (if any) are correct"""

        if isinstance(title_guess, list):
            title_guesses = title_guess
        else:
            title_guesses = [title_guess] if title_guess else []

        logger.info(indent + "Guessing metadata ...")

        title = None
        while title_guesses and not title:
            title_guess = title_guesses.pop(0)
            if title_guess:  # it might be None by mistake
                if not cautious or yesno(indent + f'Is this the document title: "{str(title_guess)}"?'):
                    title = str(title_guess)
        if not title:
            title = input(indent + "Enter document title: ")

        metadata = {"title": title}

        date_pub = None
        if date_guess:
            if not cautious or yesno(indent + f"Is this the publication date: {date_guess}?"):
                date_pub = date_guess

        if date_guess or (force_date and not date_pub):

            while not metadata.get("date_pub"):
                if type(date_pub) is date:
                    metadata["date_pub"] = date_pub
                else:
                    try:
                        metadata["date_pub"] = datetime.strptime(date_pub, "%Y-%m-%d")
                    except Exception:
                        date_pub = input(indent + "Enter publication date (yyyy-mm-dd): ")

        return metadata
