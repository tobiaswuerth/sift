from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

from sift.api.described import FIELD_LENGTH, LISTED, NAME_LENGTH, PATH_LENGTH, Bounds, Composed, Help
from sift.api.requests import Request
from sift.api.search import MOST_HITS, Options, Unsearched

NOTE = 'a note: its id, as sift get takes it, or its name as links write it, e.g. "Acme AG"'
LINKS_K = "at most this many links each way"
DOCUMENT = 'a document: its id, as sift get takes it, or a note\'s name as links write it, e.g. "Acme AG"'
ALSO = "front matter fields whose values name the note too, comma-separated, e.g. emails"


class Direction(StrEnum):
    OUT = "out"
    IN = "in"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class ShowLinks(Request):
    id: Annotated[str, Help(NOTE), PATH_LENGTH]
    direction: Annotated[Direction, Help("out (default): the links it holds; in: those into it; or both")] = (
        Direction.OUT
    )
    depth: Annotated[int, Help("2: also the notes one link further"), Bounds(1, 2)] = 1
    field: Annotated[str | None, Help("only links in this front matter field"), FIELD_LENGTH] = None
    k: Annotated[int, Help(LINKS_K), Bounds(1, MOST_HITS)] = 50


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
    unsearched: tuple[Unsearched, ...] = ()


@dataclass(frozen=True, slots=True)
class ListUnresolved(Request):
    collections: Annotated[
        tuple[str, ...],
        Help("this collection; several: each of them (default: every active one)"),
        LISTED,
        NAME_LENGTH,
    ] = ()
    k: Annotated[int, Help("at most this many targets"), Bounds(1, MOST_HITS)] = 50


@dataclass(frozen=True, slots=True)
class Target:
    target: str
    documents: int
    ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Unresolved:
    targets: tuple[Target, ...]
    unsearched: tuple[Unsearched, ...] = ()


@dataclass(frozen=True, slots=True)
class FindMentions(Options):
    id: Annotated[str, Help(NOTE), PATH_LENGTH]
    also: Annotated[tuple[str, ...], Help(ALSO), Composed(), LISTED, FIELD_LENGTH] = ()


@dataclass(frozen=True, slots=True)
class FindSimilar(Options):
    id: Annotated[str, Help(DOCUMENT), PATH_LENGTH]
