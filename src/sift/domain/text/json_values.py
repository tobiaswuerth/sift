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


class NestedTooDeeplyError(ValueError):
    pass


def _number(text: str) -> float | str:
    number = float(text)
    return number if math.isfinite(number) else text


def _integer(text: str) -> int | str:
    try:
        return int(text)
    except ValueError:
        return text


def deeper_than(text: str, most: int) -> bool:
    if text.count("[") + text.count("{") <= most:
        return False
    level, in_string, escaped = 0, False, False
    for character in text:
        if escaped:
            escaped = False
        elif in_string:
            escaped, in_string = character == "\\", character != '"'
        elif character == '"':
            in_string = True
        elif character in "[{":
            level += 1
            if level > most:
                return True
        elif character in "]}":
            level -= 1
    return False


def json_value(text: str) -> FieldValue:
    if deeper_than(text, MOST_DEPTH):
        raise NestedTooDeeplyError(f"nested more than {MOST_DEPTH} deep")
    value = cast("FieldValue", json.loads(text, parse_constant=str, parse_float=_number, parse_int=_integer))
    return without_lone_surrogates(value) if ESCAPED_SURROGATE.search(text) else value


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


def _cleaned(value: FieldValue) -> FieldValue:
    if isinstance(value, str):
        return LONE_SURROGATE.sub(REPLACEMENT, value)
    if isinstance(value, Mapping):
        return {LONE_SURROGATE.sub(REPLACEMENT, key): item for key, item in value.items()}
    if isinstance(value, Sequence):
        return list(value)
    return value


def without_lone_surrogates(value: FieldValue) -> FieldValue:
    top = _cleaned(value)
    todo = [top]
    while todo:
        held = todo.pop()
        if isinstance(held, dict):
            for key, item in held.items():
                held[key] = cleaned = _cleaned(item)
                todo.append(cleaned)
        elif isinstance(held, list):
            for at, item in enumerate(held):
                held[at] = cleaned = _cleaned(item)
                todo.append(cleaned)
    return top
