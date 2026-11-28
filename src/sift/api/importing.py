from dataclasses import dataclass
from typing import Annotated, Literal

from sift.api.described import LISTED, NAME_LENGTH, PATH_LENGTH, AbsolutePath, Help
from sift.api.indexing import Skipped
from sift.api.requests import Request

FROM = "the home of sift 1.x to import (default: this home, where sift 1.x kept its own)"
NAMED = "import this collection; several: each of them (default: every one)"


@dataclass(frozen=True, slots=True)
class ImportHome(Request):
    home: Annotated[str | None, Help(FROM), AbsolutePath(), PATH_LENGTH] = None
    collections: Annotated[tuple[str, ...], Help(NAMED), LISTED, NAME_LENGTH] = ()


@dataclass(frozen=True, slots=True)
class Brought:
    store: str
    documents_before: int
    documents: int
    passages_before: int
    passages: int
    vectors_before: int
    vectors: int
    model: str | None
    state: Literal["imported"] = "imported"


@dataclass(frozen=True, slots=True)
class CollectionImported:
    name: str
    outcome: Brought | Skipped


@dataclass(frozen=True, slots=True)
class HomeImported:
    home: str
    collections: tuple[CollectionImported, ...]
