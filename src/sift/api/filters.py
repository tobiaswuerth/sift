import json
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import cast

from sift.api.errors import InvalidRequestError

type FilterValue = str | int | float | bool

WRITTEN = re.compile(r"(\w[^=<>!]*)(>=|<=|!=|=|<|>)(.*)", re.DOTALL)
FILTER_REMEDY = "write KEY=VALUE, KEY!=VALUE, or a range such as created>=2024-06"


class Operator(StrEnum):
    ANY = "="
    NONE = "!="
    BELOW = "<"
    AT_MOST = "<="
    ABOVE = ">"
    AT_LEAST = ">="


SETS = frozenset({Operator.ANY, Operator.NONE})


@dataclass(frozen=True, slots=True)
class Filter:
    field: str
    operator: Operator
    values: tuple[FilterValue, ...]


def _refused(text: str, problem: str) -> InvalidRequestError:
    return InvalidRequestError(f"the filter {text!r} is no KEY=VALUE: {problem}", FILTER_REMEDY)


def _scalar(value: object) -> bool:
    return isinstance(value, str | int | float)


def _values(text: str, value: str) -> tuple[FilterValue, ...]:
    try:
        read = cast("object", json.loads(value))
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
    values = _values(text, value)
    if operator not in SETS and len(values) > 1:
        raise _refused(text, "a range takes one value")
    return Filter(field, operator, values)


def gathered(filters: tuple[Filter, ...]) -> tuple[Filter, ...]:
    held: dict[tuple[str, Operator] | int, Filter] = {}
    for at, each in enumerate(filters):
        key = (each.field, each.operator) if each.operator in SETS else at
        before = held.get(key)
        held[key] = each if before is None else replace(before, values=(*before.values, *each.values))
    return tuple(held.values())
