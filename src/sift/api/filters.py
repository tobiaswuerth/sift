import json
import math
import re
import unicodedata
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import cast

from sift.api.errors import InvalidRequestError

type FilterValue = str | int | float | bool

MOST_BRACKETS = 64
MOST_INTEGER = 2**63
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

    def __post_init__(self) -> None:
        if not self.field:
            raise _wrong("a filter names no field")
        if not self.values:
            raise _wrong(f"the filter on {self.field!r} has no value")
        if self.operator not in SETS and len(self.values) > 1:
            raise _wrong(f"the range on {self.field!r} takes one value, not {len(self.values)}")
        for value in self.values:
            if isinstance(value, int) and not isinstance(value, bool) and not -MOST_INTEGER <= value < MOST_INTEGER:
                raise _wrong(f"the filter on {self.field!r}: {value} is too large")
            if isinstance(value, float) and not math.isfinite(value):
                raise _wrong(f"the filter on {self.field!r}: {value} is no finite number")


def _wrong(problem: str) -> InvalidRequestError:
    return InvalidRequestError(problem, FILTER_REMEDY, "filters")


def _refused(text: str, problem: str) -> InvalidRequestError:
    return InvalidRequestError(f"the filter {text!r} is no KEY=VALUE: {problem}", FILTER_REMEDY)


def _scalar(value: object) -> bool:
    return isinstance(value, str | int | float)


def _json(value: str, *, as_written: bool) -> object:
    if as_written:
        return cast("object", json.loads(value, parse_float=str, parse_int=str, parse_constant=str))
    return cast("object", json.loads(value))


def _values(text: str, value: str, operator: Operator) -> tuple[FilterValue, ...]:
    if value.count("[") + value.count("{") > MOST_BRACKETS:
        return (value,)
    try:
        read = _json(value, as_written=operator in SETS)
    except ValueError:
        return (value,)
    held = cast("list[object]", read) if isinstance(read, list) else [read]
    if not held or not all(_scalar(each) for each in held):
        raise _refused(text, "a value is text, a number, true or false, or a list of them")
    return tuple(each for each in held if isinstance(each, str | int | float))


def written(text: str) -> Filter:
    found = WRITTEN.fullmatch(unicodedata.normalize("NFC", text))
    if found is None:
        raise _refused(text, "it names no field and operator")
    field, operator, value = found[1].strip(), Operator(found[2]), found[3].strip()
    if not value:
        raise _refused(text, "it gives no value")
    return Filter(field, operator, _values(text, value, operator))


def gathered(filters: tuple[Filter, ...]) -> tuple[Filter, ...]:
    held: dict[tuple[str, Operator] | int, Filter] = {}
    for at, each in enumerate(filters):
        key = (each.field, each.operator) if each.operator in SETS else at
        before = held.get(key)
        held[key] = each if before is None else replace(before, values=(*before.values, *each.values))
    return tuple(held.values())
