import difflib

from enum import EnumType
from typing import List

import numpy as np

from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnableLambda


def most_similar_string(string: str, strings: List[str]) -> str:
    """Return the string in a list of strings which is most similar to a given string"""

    def string_similarity(string: str, other: str) -> float:
        return difflib.SequenceMatcher(
            a=string, b=other
        ).quick_ratio()  # real_quick_ratio gives results which are far too dissimilar

    similarities = [string_similarity(string, other) for other in strings]
    idx = np.argmax(similarities)
    return strings[idx]


def get_enum_closest_match(string: str, enum_type: EnumType) -> str:
    """Return the enum in enum type which is most similar to a given string"""

    if string in enum_type.__members__:
        return enum_type(string)

    else:
        strings = list(enum_type.__members__.keys())
        return enum_type(most_similar_string(string, strings))


def force_enum_parser_single(enum_type: EnumType) -> RunnableLambda:
    """Return a parser which forces LLM output into an enum"""

    parser_if_not_null = RunnableLambda(lambda x: get_enum_closest_match(x, enum_type))

    return RunnableBranch((lambda x: x != "NULL", parser_if_not_null), (lambda *args, **kwargs: None))


def force_enum_parser_multi(enum_type: EnumType) -> RunnableLambda:
    """Return a parser which forces LLM output into a list of enums"""

    single_term_parser = force_enum_parser_single(enum_type)

    to_string = RunnableLambda(lambda x: x if isinstance(x, str) else x.content)  # x may be a string or message
    split_string = RunnableLambda(lambda string: [s.strip() for s in string.split(",")])
    parse = RunnableLambda(
        lambda strings: [s for s in [single_term_parser.invoke(string) for string in strings] if s is not None]
    )

    return to_string | split_string | parse
