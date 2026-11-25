from dataclasses import dataclass

from sift.api.errors import Problem, SiftError


@dataclass(frozen=True, slots=True)
class Notice:
    code: Problem
    message: str
    remedy: str | None
    collection: str | None = None
    id: str | None = None


def notice_of(error: SiftError, collection: str | None = None, id_: str | None = None) -> Notice:
    return Notice(error.code, error.message, error.remedy, collection, id_)
