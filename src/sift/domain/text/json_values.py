import json
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import cast

from sift.api.fields import Fields, FieldValue

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


def _changed(value: FieldValue, change: Callable[[str], str]) -> FieldValue:
    if isinstance(value, str):
        return change(value)
    if isinstance(value, Mapping):
        return {change(key): item for key, item in value.items()}
    if isinstance(value, Sequence):
        return list(value)
    return value


def strings_changed(value: FieldValue, change: Callable[[str], str]) -> FieldValue:
    top = _changed(value, change)
    todo = [top]
    while todo:
        held = todo.pop()
        if isinstance(held, dict):
            for key, item in held.items():
                held[key] = changed = _changed(item, change)
                todo.append(changed)
        elif isinstance(held, list):
            for at, item in enumerate(held):
                held[at] = changed = _changed(item, change)
                todo.append(changed)
    return top


def _whole_characters(text: str) -> str:
    return LONE_SURROGATE.sub(REPLACEMENT, text)


def without_lone_surrogates(value: FieldValue) -> FieldValue:
    return strings_changed(value, _whole_characters)


def _composed(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def composed_names(names: Iterable[str]) -> dict[str, str]:
    given = list(names)
    taken = {name for name in given if _composed(name) == name}
    renamed: dict[str, str] = {}
    for name in given:
        renamed[name] = name if _composed(name) in taken else _composed(name)
        taken.add(renamed[name])
    return renamed


def composed_fields(fields: Fields) -> Fields:
    renamed = composed_names(fields)
    return {renamed[name]: strings_changed(value, _composed) for name, value in fields.items()}
