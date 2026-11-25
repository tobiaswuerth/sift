from dataclasses import dataclass
from typing import Annotated, Literal

from sift.api.described import LISTED, NAME_LENGTH, PATH_LENGTH, Bounds, Help
from sift.api.requests import Request

MOST_FILES = 10_000
FILES = (
    "read only this file now, or drop it if gone, whatever its date: its id (collection/path), or its absolute "
    "path; several: each of them"
)


@dataclass(frozen=True, slots=True)
class IndexCollections(Request):
    names: Annotated[
        tuple[str, ...], Help("the collections to index (default: every active one)"), LISTED, NAME_LENGTH
    ] = ()
    files: Annotated[
        tuple[str, ...],
        Help(FILES),
        Bounds(0, MOST_FILES),
        PATH_LENGTH,
    ] = ()


@dataclass(frozen=True, slots=True)
class FileIndexed:
    id: str
    state: Literal["added", "changed", "unchanged", "removed", "missing", "unreadable", "left out"]
    note: str | None = None


@dataclass(frozen=True, slots=True)
class Indexed:
    files: int
    added: int
    changed: int
    removed: int
    unreadable: int
    state: Literal["indexed"] = "indexed"


@dataclass(frozen=True, slots=True)
class Skipped:
    message: str
    remedy: str | None
    state: Literal["skipped"] = "skipped"


@dataclass(frozen=True, slots=True)
class CollectionIndexed:
    name: str
    outcome: Indexed | Skipped
    named: tuple[FileIndexed, ...] = ()


@dataclass(frozen=True, slots=True)
class IndexResult:
    collections: tuple[CollectionIndexed, ...]
