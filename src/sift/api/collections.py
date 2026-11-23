from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from sift.api.described import Help, Place


class TitleFrom(StrEnum):
    HEADING = "heading"
    NAME = "name"


FOLLOW_LINKS = (
    "read the files that are links to outside the folder too, which are left out otherwise: a link may lead to any "
    "file of this machine"
)
TITLE_FROM = (
    "a note's title, unless its front matter sets one: its first heading, or its file name as Obsidian does "
    "(default: file names in an Obsidian vault, else headings)"
)


@dataclass(frozen=True, slots=True)
class AddCollection:
    folder: Annotated[str, Help("the folder whose files the collection holds"), Place()]
    name: Annotated[str | None, Help("its name (default: the folder's)")] = None
    store: Annotated[
        str | None,
        Help(
            "where its index goes: a folder, or a file ending in .sqlite (default: the folder collections in SIFT_HOME)"
        ),
        Place(),
    ] = None
    title_from: Annotated[TitleFrom | None, Help(TITLE_FROM)] = None
    nested: Annotated[
        bool,
        Help("add a folder inside a collection, or one that holds a collection, though their files are indexed twice"),
    ] = False
    follow_links: Annotated[bool, Help(FOLLOW_LINKS)] = False


@dataclass(frozen=True, slots=True)
class CollectionAdded:
    name: str
    folder: str
    store: str
    title_from: TitleFrom
    vault: bool
    follow_links: bool


@dataclass(frozen=True, slots=True)
class ShowStatus:
    names: Annotated[tuple[str, ...], Help("the collections to show (default: all)")] = ()
    unreadable: Annotated[bool, Help("list the files that could not be read, and why")] = False


@dataclass(frozen=True, slots=True)
class UnreadableFile:
    path: str
    problem: str


@dataclass(frozen=True, slots=True)
class Counted:
    documents: int
    chunks: int
    stored_bytes: int
    unreadable: int
    unreadable_files: tuple[UnreadableFile, ...]
    links: int = 0
    unresolved: int = 0
    state: Literal["counted"] = "counted"


@dataclass(frozen=True, slots=True)
class Unavailable:
    message: str
    remedy: str | None
    state: Literal["unavailable"] = "unavailable"


@dataclass(frozen=True, slots=True)
class CollectionStatus:
    name: str
    folder: str
    store: str
    contents: Counted | Unavailable


@dataclass(frozen=True, slots=True)
class Status:
    home: str
    collections: tuple[CollectionStatus, ...]
