import json
import re
from typing import cast

from sift.api.errors import InvalidRequestError
from sift.api.filters import Filter, FilterValue, Operator

MOST_BRACKETS = 64
WRITTEN = re.compile(r"(\w[^=<>!]*)(>=|<=|!=|=|<|>)(.*)", re.DOTALL)
WRITTEN_SYNTAX = "written KEY=VALUE, KEY!=VALUE, or a range such as created>=2024-06"
WRITTEN_REMEDY = "write KEY=VALUE, KEY!=VALUE, or a range such as created>=2024-06"


def listed(text: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in text.split(",")) if text.strip() else ()


def _refused(text: str, problem: str) -> InvalidRequestError:
    return InvalidRequestError(f"the filter {text!r} is no KEY=VALUE: {problem}", WRITTEN_REMEDY, "filters")


def _scalar(value: object) -> bool:
    return isinstance(value, str | int | float)


def _values(text: str, value: str) -> tuple[FilterValue, ...]:
    if value.count("[") + value.count("{") > MOST_BRACKETS:
        return (value,)
    try:
        read = cast("object", json.loads(value, parse_float=str, parse_int=str, parse_constant=str))
    except ValueError:
        return (value,)
    held = cast("list[object]", read) if isinstance(read, list) else [read]
    if not held or not all(_scalar(each) for each in held):
        raise _refused(text, "a value is text, a number, true or false, or a list of them")
    return tuple(each for each in held if isinstance(each, str | int | float))


def written(text: str) -> Filter:
    found = WRITTEN.fullmatch(text)
    if found is None:
        raise _refused(text, "it names no field and operator")
    field, operator, value = found[1].strip(), Operator(found[2]), found[3].strip()
    if not value:
        raise _refused(text, "it gives no value")
    return Filter(field, operator, _values(text, value))
