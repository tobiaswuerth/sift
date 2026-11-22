from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from sift.api.described import Help, Place


class TitleFrom(StrEnum):
    HEADING = "heading"
    NAME = "name"


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


@dataclass(frozen=True, slots=True)
class CollectionAdded:
    name: str
    folder: str
    store: str
    title_from: TitleFrom
    vault: bool


@dataclass(frozen=True, slots=True)
class ShowStatus:
    pass


@dataclass(frozen=True, slots=True)
class Counted:
    documents: int
    chunks: int
    stored_bytes: int
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
