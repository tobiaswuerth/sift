from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

from sift.api.described import Help, Short
from sift.api.search import Unsearched

NOTE = 'a note: its id, as sift get takes it, or its name as links write it, e.g. "Acme AG"'
LINKS_K = "at most this many links each way (default: 50)"


class Direction(StrEnum):
    OUT = "out"
    IN = "in"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class ShowLinks:
    id: Annotated[str, Help(NOTE)]
    direction: Annotated[Direction, Help("out (default): the links it holds; in: those into it; or both")] = (
        Direction.OUT
    )
    depth: Annotated[int, Help("2: also the notes one link further (default: 1)")] = 1
    field: Annotated[str | None, Help("only links in this front matter field")] = None
    k: Annotated[int, Help(LINKS_K), Short("-k")] = 50


@dataclass(frozen=True, slots=True)
class LinkOut:
    line: int
    target: str
    field: str | None
    fragment: str | None
    via: str
    id: str | None
    title: str | None
    ambiguous: tuple[str, ...]
    through: str | None


@dataclass(frozen=True, slots=True)
class LinkingLine:
    line: int
    field: str | None
    text: str


@dataclass(frozen=True, slots=True)
class LinkIn:
    id: str
    title: str
    lines: tuple[LinkingLine, ...]
    through: str | None


@dataclass(frozen=True, slots=True)
class Links:
    id: str
    title: str
    out: tuple[LinkOut, ...] | None
    into: tuple[LinkIn, ...] | None


@dataclass(frozen=True, slots=True)
class ListUnresolved:
    collection: Annotated[str | None, Help("this collection alone (default: every active one)"), Short("-c")] = None
    k: Annotated[int, Help("at most this many targets (default: 50)"), Short("-k")] = 50


@dataclass(frozen=True, slots=True)
class Target:
    target: str
    documents: int
    ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Unresolved:
    targets: tuple[Target, ...]
    unsearched: tuple[Unsearched, ...] = ()
