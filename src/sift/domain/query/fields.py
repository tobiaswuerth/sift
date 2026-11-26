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
UNPADDED = re.compile(r"(?<=-)([0-9])(?![0-9])")

MOST_EXACT_DIGITS = 300

type Key = tuple[int, int | float, str]


def checked(sort: str | None) -> None:
    if sort is not None and not sort.removeprefix("-"):
        raise InvalidRequestError(f"{sort!r} names no field to sort by", SORT_REMEDY, "sort")


def picked(held: Fields, asked: Sequence[str]) -> Fields:
    if EVERY in asked:
        return {key: value for key, value in held.items() if key != TITLE}
    return {key: held[key] for key in asked if key in held}


def _whole(digits: str) -> int | float:
    return int(digits) if len(digits) <= MOST_EXACT_DIGITS else float(digits)


def number_read(value: FieldValue) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    if not isinstance(value, str):
        return value
    text = value.strip(" ")
    if SIGNED_WHOLE.fullmatch(text):
        return _whole(text)
    if SIGNED_DECIMAL.fullmatch(text):
        return float(text)
    dated = DATED.match(text)
    return _whole(dated.group(1)) if dated else None


def _text(value: FieldValue) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _dated(text: str) -> str:
    return UNPADDED.sub(r"0\1", text.replace("/", "-")) if DATED.match(text.strip(" ")) else text


def _key(value: FieldValue) -> Key:
    first = (value[0] if value else "") if isinstance(value, list) else value
    number = number_read(first)
    if number is not None:
        return (0, number, _dated(first) if isinstance(first, str) else _text(first))
    if isinstance(first, bool):
        return (1, int(first), "")
    return (2, 0, _text(first))


def sorted_by(sort: str) -> str:
    return sort.removeprefix("-")


def unheld(each: Sequence[Fields], asked: Sequence[str]) -> list[str]:
    return [name for name in asked if name != EVERY and not any(name in held for held in each)]


def unsorted(each: Sequence[Fields], sort: str | None) -> str | None:
    if sort is None or any(held.get(sorted_by(sort)) is not None for held in each):
        return None
    return sorted_by(sort)


def order(each: Sequence[Fields], sort: str) -> list[int]:
    field, descending = sorted_by(sort), sort.startswith("-")
    valued = [at for at, held in enumerate(each) if held.get(field) is not None]
    valued.sort(key=lambda at: _key(each[at][field]), reverse=descending)
    return valued + [at for at, held in enumerate(each) if held.get(field) is None]
