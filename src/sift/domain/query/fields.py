from collections.abc import Sequence

from sift.api.errors import InvalidRequestError
from sift.api.fields import Fields, FieldValue
from sift.domain.query.filters import DECIMAL, WHOLE

EVERY = "*"
TITLE = "title"
SORT_REMEDY = "name a front matter field, e.g. created, or -created for the newest first"

type Key = tuple[int, int | float, str]


def checked(sort: str | None) -> None:
    if sort is not None and not sort.removeprefix("-"):
        raise InvalidRequestError(f"{sort!r} names no field to sort by", SORT_REMEDY, "sort")


def picked(held: Fields, asked: Sequence[str]) -> Fields:
    if EVERY in asked:
        return {key: value for key, value in held.items() if key != TITLE}
    return {key: held[key] for key in asked if key in held}


def _key(value: FieldValue) -> Key:
    if isinstance(value, int | float):
        return (0, value, "")
    written = str(value)
    if WHOLE.fullmatch(written):
        return (0, int(written), "")
    return (0, float(written), "") if DECIMAL.fullmatch(written) else (1, 0.0, written)


def order(each: Sequence[Fields], sort: str) -> list[int]:
    field, descending = sort.removeprefix("-"), sort.startswith("-")
    valued = [at for at, held in enumerate(each) if held.get(field) is not None]
    valued.sort(key=lambda at: _key(each[at][field]), reverse=descending)
    return valued + [at for at, held in enumerate(each) if held.get(field) is None]
