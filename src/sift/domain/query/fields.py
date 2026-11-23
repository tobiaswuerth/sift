from collections.abc import Sequence

from sift.api.errors import InvalidRequestError
from sift.api.fields import Fields, FieldValue

EVERY = "*"
TITLE = "title"
FIELDS_REMEDY = 'name front matter fields parted by commas, e.g. --fields url,created, or "*" for all'
SORT_REMEDY = "name a front matter field, e.g. --sort created, or --sort=-created for the newest first"

type Key = tuple[int, int | float, str]


def checked(fields: Sequence[str], sort: str | None) -> None:
    if any(not name for name in fields):
        raise InvalidRequestError(f"--fields {','.join(fields)!r} names no field between two commas", FIELDS_REMEDY)
    if sort is not None and not sort.removeprefix("-"):
        raise InvalidRequestError(f"--sort {sort!r}: the sort names no field", SORT_REMEDY)


def picked(held: Fields, asked: Sequence[str]) -> Fields:
    if EVERY in asked:
        return {key: value for key, value in held.items() if key != TITLE}
    return {key: held[key] for key in asked if key in held}


def _key(value: FieldValue) -> Key:
    if isinstance(value, int | float):
        return (0, value, "")
    return (1, 0.0, str(value))


def order(each: Sequence[Fields], sort: str) -> list[int]:
    field, descending = sort.removeprefix("-"), sort.startswith("-")
    valued = [at for at, held in enumerate(each) if held.get(field) is not None]
    valued.sort(key=lambda at: _key(each[at][field]), reverse=descending)
    return valued + [at for at, held in enumerate(each) if held.get(field) is None]
