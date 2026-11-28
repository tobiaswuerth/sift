import json
import math
import re
import unicodedata
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import cast

from sift.api.described import FIELD_LENGTH, TEXT_LENGTH
from sift.api.errors import InvalidRequestError
from sift.api.requests import HALF_PAIR_REMEDY, holds_half_a_pair, shown_value

type FilterValue = str | int | float | bool

MOST_INTEGER = 2**63
MOST_VALUES = 100_000
WHOLE = re.compile("-?[0-9]+")
FILTER_REMEDY = (
    "give a field, an operator (=, !=, <, <=, >, >=) and values, "
    'e.g. {"field": "year", "operator": ">=", "values": [2024]}'
)


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
        field, operator = cast("object", self.field), cast("object", self.operator)
        if not isinstance(operator, Operator):
            raise _wrong(f"the operator {operator!r} is no Operator: give one of {', '.join(Operator)}", "operator")
        if not isinstance(field, str) or not field:
            raise _wrong("a filter names no field", "field")
        if holds_half_a_pair(field) or holds_half_a_pair(self.values):
            raise InvalidRequestError("a filter holds half a surrogate pair", HALF_PAIR_REMEDY, "filters")
        if not self.values:
            raise _wrong(f"the filter on {self.field!r} has no value", "values")
        if self.operator not in SETS and len(self.values) > 1:
            raise _wrong(f"the range on {self.field!r} takes one value, not {len(self.values)}", "values")
        _refuse_too_long(field, cast("tuple[object, ...]", self.values))
        object.__setattr__(self, "field", unicodedata.normalize("NFC", field))
        values = tuple(
            _meant(value, self.field, ranged=self.operator not in SETS)
            for value in cast("tuple[object, ...]", self.values)
        )
        object.__setattr__(self, "values", values)
        for value in values:
            if not isinstance(value, str | int | float):
                raise _wrong(f"the filter on {self.field!r}: {value!r} is no text, number, true or false", "values")
            if isinstance(value, int) and not isinstance(value, bool) and not -MOST_INTEGER <= value < MOST_INTEGER:
                size = "large" if value > 0 else "small"
                raise _wrong(f"the filter on {self.field!r}: {shown_value(value)} is too {size}", "values")
            if isinstance(value, float) and not math.isfinite(value):
                raise _wrong(f"the filter on {self.field!r}: {value} is no finite number", "values")


def _refuse_too_long(field: str, values: tuple[object, ...]) -> None:
    if len(field) > FIELD_LENGTH.high:
        raise _wrong(f"a filter's field is {len(field):,} characters long, more than {FIELD_LENGTH.high:,}", "field")
    if len(values) > MOST_VALUES:
        raise _wrong(f"the filter on {field[:40]!r} has {len(values):,} values, more than {MOST_VALUES:,}", "values")
    longest = max((len(each) for each in values if isinstance(each, str)), default=0)
    if longest > TEXT_LENGTH.high:
        raise _wrong(
            f"the filter on {field[:40]!r} has a value {longest:,} characters long, more than {TEXT_LENGTH.high:,}",
            "values",
        )


def _no_constant(name: str) -> object:
    raise ValueError(name)


def _number_in(text: str, field: str) -> object:
    try:
        read = cast("object", json.loads(text, parse_constant=_no_constant))
    except ValueError:
        if WHOLE.fullmatch(text):
            size = "small" if text.startswith("-") else "large"
            raise _wrong(f"the filter on {field!r}: {text[:40]}... is too {size}", "values") from None
        return text
    if isinstance(read, float) and not math.isfinite(read):
        raise _wrong(f"the filter on {field!r}: {text[:40]} is no finite number", "values")
    return read if isinstance(read, int | float) and not isinstance(read, bool) else text


def _meant(value: object, field: str, *, ranged: bool) -> object:
    if not isinstance(value, str):
        return value
    text = unicodedata.normalize("NFC", value)
    return _number_in(text, field) if ranged else text


def _wrong(problem: str, part: str) -> InvalidRequestError:
    return InvalidRequestError(problem, FILTER_REMEDY, f"filters.{part}")


def gathered(filters: tuple[Filter, ...]) -> tuple[Filter, ...]:
    held: dict[tuple[str, Operator] | int, Filter] = {}
    for at, each in enumerate(filters):
        key = (each.field, each.operator) if each.operator in SETS else at
        before = held.get(key)
        held[key] = each if before is None else replace(before, values=(*before.values, *each.values))
    return tuple(held.values())
