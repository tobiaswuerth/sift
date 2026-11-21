"""Conditions on front matter, as SQL: how a filter's value compares with a value a document holds, the `value` of
SQLite's json_each, and how a field is addressed. Text alone: what runs it, and against which documents, is the
search's (see search.Index._where)."""

import json
import math
import re
import sqlite3
from typing import Any, TypeGuard, cast

Value = str | int | float  # what a filter compares with; yes and no are numbers here, as Python has them

ESCAPES = sqlite3.sqlite_version_info >= (3, 45)  # whether SQLite reads the escapes of a name in a JSON path

SQLITE_INT = 2**63  # SQLite integers are 64-bit
OPERATORS = ("=", "<", "<=", ">", ">=", "not")
# The values a number compares with (`type` and `value` of json_each): numbers, yes and no as 1 and 0, and text that
# starts with a number, as a date does with its year. SQLite casts other text, "unknown", to 0, and a list to text;
# text compares with text and numbers alone, not with a list's JSON.
NUMBER_FIRST = (
    "(type IN ('integer', 'real', 'true', 'false') OR (type = 'text' AND (ltrim(value) GLOB '[0-9]*'"
    " OR ltrim(value) GLOB '[-+.][0-9]*' OR ltrim(value) GLOB '[-+].[0-9]*')))"
)


def each(value: object) -> list[Any]:
    """The values of a front-matter field or a filter: a list's, or the one."""
    return value if _is_list(value) else [value]


def conditions(value: object) -> dict[Any, Any]:
    """A filter's conditions, {operator: value}: a value alone is {"=": value}."""
    return value if _is_dict(value) else {"=": value}


def _is_list(value: object) -> TypeGuard[list[Any]]:  # parsed from JSON, or a caller's: it may hold anything
    return isinstance(value, list)


def _is_dict(value: object) -> TypeGuard[dict[Any, Any]]:
    return isinstance(value, dict)


def comparison(key: str, op: str, value: object) -> tuple[str, list[Value]]:
    """SQL comparing a front-matter value (`value` of json_each) with a filter value, and its arguments. For "=" and
    "not", a list stands for any of its values."""
    if op not in OPERATORS:
        raise ValueError(f"filter {key!r}: unknown operator {op!r}, use one of {', '.join(OPERATORS)}")
    several = isinstance(value, list) and op in ("=", "not")
    given: list[object] = cast("list[object]", value) if several else [value]
    if not given:
        raise ValueError(f"filter {key!r} needs at least one value")
    values: list[Value] = []
    for one in given:
        if not isinstance(one, str | int | float):
            raise ValueError(f"filter {key!r} needs a string or number, or for = and not a list of them, got {one!r}")
        if isinstance(one, int) and not -SQLITE_INT <= one < SQLITE_INT:
            raise ValueError(f"filter {key!r}: {one} is too large")
        if isinstance(one, float) and not math.isfinite(one):
            raise ValueError(f"filter {key!r}: {one} is no finite number")
        values.append(one)
    if op in ("=", "not"):  # as one JSON array: a variable each, a long list passed SQLite's limit of them
        values = list(dict.fromkeys(form for one in values for form in forms(one)))
        return "value IN (SELECT value FROM json_each(?))", [json.dumps(values)]
    if isinstance(value, str):  # at the precision given: "2024-12-31" is <= "2024", and not > "2024"
        return f"type IN ('text', 'integer', 'real') AND substr(value, 1, ?) {op} ?", [len(value), value]
    return f"{NUMBER_FIRST} AND CAST(value AS REAL) {op} ?", values  # SQLite reads "2024-06-15" as 2024


def forms(value: Value) -> list[Value]:
    """A value to match, in each form front matter may hold it: as given, and a text as the number or the yes or no it
    reads as, a number or a yes or no as the text that writes it. "2024" is a number in one document and text in
    another, as JSON Lines write it, and nobody asking knows which; "007" stays text too, where it is an agent's
    number."""
    if isinstance(value, bool):
        return [value, "true" if value else "false"]
    if not isinstance(value, str):
        return [value, json.dumps(value)]
    if value in ("true", "false"):
        return [value, value == "true"]
    if re.fullmatch(r"-?[0-9]+", value) and -SQLITE_INT <= int(value) < SQLITE_INT:
        return [value, int(value)]
    return [value, float(value)] if re.fullmatch(r"-?[0-9]+\.[0-9]+", value) else [value]


def json_path(name: object, what: str = "filter field") -> str:
    """A front-matter field name, as SQLite's JSON functions address it: any name, written as a JSON string, which
    SQLite reads with its escapes, so that a backslash or a double quote in it is the name's own. Its letters as they
    are, as indexing stores front matter: SQLite before 3.45 compares the name with the stored one byte for byte, and
    ends it at its first double quote, so there a name that holds one is refused."""
    if not (isinstance(name, str) and name):
        raise ValueError(f"invalid {what} {name!r}")
    if '"' in name and not ESCAPES:
        raise ValueError(
            f"invalid {what} {name!r}: SQLite reads a double quote in a field's name from 3.45 on, and this is "
            f"{sqlite3.sqlite_version}"
        )
    return "$." + json.dumps(name, ensure_ascii=False)
