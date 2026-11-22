import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import cast

from sift.api.fields import FieldValue

MOST_DEPTH = 999
ESCAPED_SURROGATE = re.compile(r"\\u[dD][89a-fA-F]")
LONE_SURROGATE = re.compile("[\ud800-\udfff]")
REPLACEMENT = "\N{REPLACEMENT CHARACTER}"


class NestedTooDeeplyError(RecursionError):
    pass


def _number(text: str) -> float | str:
    number = float(text)
    return number if math.isfinite(number) else text


def json_value(text: str) -> FieldValue:
    value = cast("FieldValue", json.loads(text, parse_constant=str, parse_float=_number))
    if text.count("[") + text.count("{") > MOST_DEPTH and depth(value) > MOST_DEPTH:
        raise NestedTooDeeplyError(f"nested more than {MOST_DEPTH} deep")
    return without_lone_surrogates(value) if ESCAPED_SURROGATE.search(text) else value


def is_list(value: FieldValue) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str)


def inside(value: FieldValue) -> Sequence[FieldValue] | None:
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    return None


def depth(value: FieldValue) -> int:
    deepest, todo = 0, [(value, 0)]
    while todo:
        item, level = todo.pop()
        if (children := inside(item)) is not None:
            deepest = max(deepest, level + 1)
            todo.extend((child, level + 1) for child in children)
    return deepest


def without_lone_surrogates(value: FieldValue) -> FieldValue:
    if isinstance(value, str):
        return LONE_SURROGATE.sub(REPLACEMENT, value)
    if isinstance(value, Mapping):
        return {LONE_SURROGATE.sub(REPLACEMENT, key): without_lone_surrogates(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [without_lone_surrogates(item) for item in value]
    return value
