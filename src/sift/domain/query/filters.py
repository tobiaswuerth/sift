import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from sift.api.errors import InvalidRequestError
from sift.api.filters import FILTER_REMEDY, SETS, Filter, FilterValue, Operator, gathered
from sift.domain.text.links import link_target

type SpecValue = str | int | float | bool | list[SpecValue] | dict[str, SpecValue] | None

SQLITE_INT = 2**63
WHOLE = re.compile(r"-?[0-9]+")
DECIMAL = re.compile(r"-?[0-9]+\.[0-9]+")
NONE_WITHIN = json.dumps([])
QUOTE_REMEDY = "name the field without a double quote, or use a Python whose SQLite is 3.47 or newer"


@dataclass(frozen=True, slots=True)
class Condition:
    path: str | None
    fields: str
    within: str | None = None
    without: str | None = None

    @property
    def lets_none(self) -> bool:
        return self.within == NONE_WITHIN


@dataclass(frozen=True, slots=True)
class LinkFilter:
    field: str
    operator: Operator
    target: str


@dataclass(frozen=True, slots=True)
class Linked:
    values: Mapping[LinkFilter, Sequence[int]] = MappingProxyType({})
    within: Sequence[int] | None = None
    without: Sequence[int] = ()


UNLINKED = Linked()


def _refused(problem: str, remedy: str = FILTER_REMEDY) -> InvalidRequestError:
    return InvalidRequestError(problem, remedy)


def _refuse_unreadable(field: str, *, reads_quotes: bool) -> None:
    if '"' in field and not reads_quotes:
        problem = f"the field {field!r} holds a double quote, which this SQLite reads in no field's name"
        raise _refused(problem, QUOTE_REMEDY)


def json_path(field: str, *, reads_quotes: bool) -> str:
    _refuse_unreadable(field, reads_quotes=reads_quotes)
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


def readable(filters: Sequence[Filter], *, reads_quotes: bool) -> None:
    for each in filters:
        _refuse_unreadable(each.field, reads_quotes=reads_quotes)


def _plain(value: object) -> bool:
    return not isinstance(value, str) or link_target(value) is None


def _held(filters: list[Filter], operator: Operator) -> SpecValue:
    asked = [each for each in filters if each.operator is operator]
    found = [form for each in asked for value in each.values if _plain(value) for form in forms(value)]
    return list(dict.fromkeys(found)) if asked else None


def _range(each: Filter) -> SpecValue:
    (value,) = each.values
    text = isinstance(value, str)
    return {"op": str(each.operator), "text": text, "len": len(value) if text else 0, "value": value}


def _linkings(each: Filter) -> list[LinkFilter]:
    targets = [link_target(value) for value in each.values if isinstance(value, str)] if each.operator in SETS else []
    return [LinkFilter(each.field, each.operator, target) for target in targets if target is not None]


def linkings(filters: Sequence[Filter]) -> tuple[LinkFilter, ...]:
    return tuple(dict.fromkeys(found for each in gathered(tuple(filters)) for found in _linkings(each)))


def _linked(filters: list[Filter], operator: Operator, linked: Mapping[LinkFilter, Sequence[int]]) -> SpecValue:
    found = [linked.get(each, ()) for one in filters if one.operator is operator for each in _linkings(one)]
    return list[SpecValue](sorted({document for documents in found for document in documents})) if found else None


def _group(
    field: str, filters: list[Filter], linked: Mapping[LinkFilter, Sequence[int]], *, reads_quotes: bool
) -> SpecValue:
    ranges = [_range(each) for each in filters if each.operator not in SETS]
    return {
        "path": json_path(field, reads_quotes=reads_quotes),
        "any": _held(filters, Operator.ANY),
        "none": _held(filters, Operator.NONE),
        "ranges": ranges,
        "linked_any": None if ranges else _linked(filters, Operator.ANY, linked),
        "linked_none": _linked(filters, Operator.NONE, linked),
    }


def condition(
    path: str | None,
    filters: Sequence[Filter],
    *,
    reads_quotes: bool,
    linked: Linked = UNLINKED,
) -> Condition | None:
    asked = gathered(tuple(filters))
    if not path and not asked and linked.within is None and not linked.without:
        return None
    groups = [
        _group(field, [each for each in asked if each.field == field], linked.values, reads_quotes=reads_quotes)
        for field in dict.fromkeys(each.field for each in asked)
    ]
    kept = None if linked.within is None else json.dumps(sorted(set(linked.within)))
    left = json.dumps(sorted(set(linked.without))) if linked.without else None
    return Condition(path or None, json.dumps(groups, ensure_ascii=False), kept, left)
