from typing import Callable
from typing import List
from typing import Tuple
from typing import Union

from dsp_nesta_brain import logger
from langdetect import detect


def first(seq: Union[List, Tuple], lambda_: Callable) -> object:
    """Return first element of seq that returns a positive result from the condition specified in lambda_"""
    return next((ele for ele in seq if lambda_(ele)), None)


def unique(seq: Union[List, Tuple]) -> List:
    """Find unique elements of a sequence and retain order"""
    seen = {}
    result = []
    for item in seq:
        if item in seen:
            continue
        seen[item] = 1
        result.append(item)
    return result


def bold(string: str) -> str:
    """Make a string bold"""
    return f"\033[1m{string}\033[0m"


def is_english(string: str) -> bool:
    """Test whether a string is in English"""
    return detect(string) == "en"


def yesno(question: str) -> bool:
    """Answer a yes/no question and return the answer as a bool"""
    instructions = " (0/1 or y/n): "
    question = f"{question} {instructions}"
    while True:
        resp = input(question)
        if resp.lower() in ["0", "1", "y", "n", ""]:
            return resp.lower() in ["1", "y"]
        else:
            logger.info("Invalid binary question input, try again: ")
