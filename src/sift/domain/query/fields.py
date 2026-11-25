import json
import re
from collections.abc import Sequence

from sift.api.errors import InvalidRequestError
from sift.api.fields import Fields, FieldValue

EVERY = "*"
TITLE = "title"
SORT_REMEDY = "name a front matter field, e.g. created, or -created for the newest first"

SIGNED_WHOLE = re.compile(r"[+-]?[0-9]+")
SIGNED_DECIMAL = re.compile(r"[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+)")
DATED = re.compile(r"([0-9]+)[-/]")

type Key = tuple[int, int | float, str]


def checked(sort: str | None) -> None:
    if sort is not None and not sort.removeprefix("-"):
        raise InvalidRequestError(f"{sort!r} names no field to sort by", SORT_REMEDY, "sort")


def picked(held: Fields, asked: Sequence[str]) -> Fields:
    if EVERY in asked:
        return {key: value for key, value in held.items() if key != TITLE}
    return {key: held[key] for key in asked if key in held}


def number_read(value: FieldValue) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    if not isinstance(value, str):
        return value
    text = value.strip(" ")
    if SIGNED_WHOLE.fullmatch(text):
        return int(text)
    if SIGNED_DECIMAL.fullmatch(text):
        return float(text)
    dated = DATED.match(text)
    return int(dated.group(1)) if dated else None


def _text(value: FieldValue) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _key(value: FieldValue) -> Key:
    first = (value[0] if value else "") if isinstance(value, list) else value
    number = number_read(first)
    if number is not None:
        return (0, number, _text(first))
    if isinstance(first, bool):
        return (1, int(first), "")
    return (2, 0, _text(first))


def order(each: Sequence[Fields], sort: str) -> list[int]:
    field, descending = sort.removeprefix("-"), sort.startswith("-")
    valued = [at for at, held in enumerate(each) if held.get(field) is not None]
    valued.sort(key=lambda at: _key(each[at][field]), reverse=descending)
    return valued + [at for at, held in enumerate(each) if held.get(field) is None]
