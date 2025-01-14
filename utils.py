from typing import Callable
from typing import List
from typing import Tuple
from typing import Union


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
