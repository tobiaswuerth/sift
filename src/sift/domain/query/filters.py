import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from sift.api.errors import InvalidRequestError
from sift.api.filters import FILTER_REMEDY, SETS, Filter, FilterValue, Operator, gathered

type SpecValue = str | int | float | bool | list[SpecValue] | dict[str, SpecValue] | None

SQLITE_INT = 2**63
WHOLE = re.compile(r"-?[0-9]+")
DECIMAL = re.compile(r"-?[0-9]+\.[0-9]+")
QUOTE_REMEDY = "name the field without a double quote, or use a Python whose SQLite is 3.47 or newer"


@dataclass(frozen=True, slots=True)
class Condition:
    path: str | None
    fields: str


def _refused(problem: str, remedy: str = FILTER_REMEDY) -> InvalidRequestError:
    return InvalidRequestError(problem, remedy)


def json_path(field: str, *, reads_quotes: bool) -> str:
    if not field:
        raise _refused("a filter names no field")
    if '"' in field and not reads_quotes:
        problem = f"the field {field!r} holds a double quote, which this SQLite reads in no field's name"
        raise _refused(problem, QUOTE_REMEDY)
    return "$." + json.dumps(field, ensure_ascii=False)


def forms(value: FilterValue) -> list[SpecValue]:
    if isinstance(value, bool):
        return [int(value), "true" if value else "false"]
    if not isinstance(value, str):
        return [value, json.dumps(value)]
    if value in ("true", "false"):
        return [value, int(value == "true")]
    if WHOLE.fullmatch(value) and -SQLITE_INT <= int(value) < SQLITE_INT:
        return [value, int(value)]
    return [value, float(value)] if DECIMAL.fullmatch(value) else [value]


def _checked(each: Filter) -> None:
    if not each.values:
        raise _refused(f"the filter on {each.field!r} has no value")
    if each.operator not in SETS and len(each.values) > 1:
        raise _refused(f"the range on {each.field!r} takes one value, not {len(each.values)}")
    for value in each.values:
        if isinstance(value, int) and not isinstance(value, bool) and not -SQLITE_INT <= value < SQLITE_INT:
            raise _refused(f"the filter on {each.field!r}: {value} is too large")
        if isinstance(value, float) and not math.isfinite(value):
            raise _refused(f"the filter on {each.field!r}: {value} is no finite number")


def _held(filters: list[Filter], operator: Operator) -> SpecValue:
    found = [form for each in filters if each.operator is operator for value in each.values for form in forms(value)]
    return list(dict.fromkeys(found)) if found else None


def _range(each: Filter) -> SpecValue:
    (value,) = each.values
    text = isinstance(value, str)
    return {"op": str(each.operator), "text": text, "len": len(value) if text else 0, "value": value}


def _group(field: str, filters: list[Filter], *, reads_quotes: bool) -> SpecValue:
    return {
        "path": json_path(field, reads_quotes=reads_quotes),
        "any": _held(filters, Operator.ANY),
        "none": _held(filters, Operator.NONE),
        "ranges": [_range(each) for each in filters if each.operator not in SETS],
    }


def condition(path: str | None, filters: Sequence[Filter], *, reads_quotes: bool) -> Condition | None:
    asked = gathered(tuple(filters))
    for each in asked:
        _checked(each)
    if not path and not asked:
        return None
    groups = [
        _group(field, [each for each in asked if each.field == field], reads_quotes=reads_quotes)
        for field in dict.fromkeys(each.field for each in asked)
    ]
    return Condition(path or None, json.dumps(groups, ensure_ascii=False))
