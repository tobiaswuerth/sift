from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal

from sift.api.described import FIELD_LENGTH, LISTED, NAME_LENGTH, PATH_LENGTH, AbsolutePath, Bounds, Help
from sift.api.notices import Notice
from sift.api.requests import Request

MOST_NAMED = 1_000


class TitleFrom(StrEnum):
    HEADING = "heading"
    NAME = "name"


AGAIN = "; given for a folder that is a collection already, it changes that collection"
FOLLOW_LINKS = (
    "read the files that are links to outside the folder too, which are left out otherwise: a link may lead to any "
    f"file of this machine{AGAIN}"
)
ID_FIELD = (
    "JSON Lines: the field holding each record's id, a text or a whole number (default: id or _id, else the "
    "line number, which changes when lines are added above)"
)
TITLE_FIELD = "JSON Lines: the field holding the title (default: title, name, headline or subject)"
TEXT_FIELDS = (
    "JSON Lines: the fields whose text a record's document holds "
    "(default: text, content, body, abstract, description or summary, else the longest string)"
)
AGENTS = f"let agents see it through sift's tools for them (default: yes){AGAIN}"
TITLE_FROM = (
    "a note's title, unless its front matter sets one: its first heading, or its file name as Obsidian does "
    f"(default: file names in an Obsidian vault, else headings){AGAIN}"
)


@dataclass(frozen=True, slots=True)
class AddCollection(Request):
    folder: Annotated[str, Help("the folder whose files the collection holds"), AbsolutePath(), PATH_LENGTH]
    name: Annotated[str | None, Help("its name (default: the folder's)"), NAME_LENGTH] = None
    store: Annotated[
        str | None,
        Help(
            "where its index goes: a folder, or a file ending in .sqlite (default: the folder collections in SIFT_HOME)"
        ),
        AbsolutePath(),
        PATH_LENGTH,
    ] = None
    title_from: Annotated[TitleFrom | None, Help(TITLE_FROM)] = None
    nested: Annotated[
        bool,
        Help("add a folder inside a collection, or one that holds a collection, though their files are indexed twice"),
    ] = False
    follow_links: Annotated[bool | None, Help(FOLLOW_LINKS)] = None
    agents: Annotated[bool | None, Help(AGENTS)] = None
    id_field: Annotated[str | None, Help(ID_FIELD), FIELD_LENGTH] = None
    title_field: Annotated[str | None, Help(TITLE_FIELD), FIELD_LENGTH] = None
    text_fields: Annotated[tuple[str, ...], Help(TEXT_FIELDS), LISTED, FIELD_LENGTH] = ()


@dataclass(frozen=True, slots=True)
class CollectionAdded:
    name: str
    folder: str
    store: str
    title_from: TitleFrom
    vault: bool
    follow_links: bool
    agents: bool = True
    state: Literal["added"] = "added"


@dataclass(frozen=True, slots=True)
class CollectionChanged:
    name: str
    title_from: TitleFrom
    follow_links: bool
    agents: bool
    retitled: int
    to_embed: int
    index_again: bool = False
    state: Literal["changed"] = "changed"


@dataclass(frozen=True, slots=True)
class RemoveCollection(Request):
    name: Annotated[str, Help("the collection to remove, with its store; its files stay where they are"), NAME_LENGTH]


@dataclass(frozen=True, slots=True)
class CollectionRemoved:
    name: str
    store: str
    documents: int | None
    problem: Notice | None
    left: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MoveCollections(Request):
    names: Annotated[
        tuple[str, ...], Help("the collections whose stores to move (default: every one)"), LISTED, NAME_LENGTH
    ] = ()
    to: Annotated[
        str, Help("a folder, or for one collection, a file ending in .sqlite"), AbsolutePath(), PATH_LENGTH
    ] = field(kw_only=True)


@dataclass(frozen=True, slots=True)
class StoreMoved:
    name: str
    old: str
    new: str
    copied: bool
    left: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StoresMoved:
    moved: tuple[StoreMoved, ...]
    already: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ActivateCollections(Request):
    names: Annotated[tuple[str, ...], Help("the collections to search again"), Bounds(1, MOST_NAMED), NAME_LENGTH]


@dataclass(frozen=True, slots=True)
class DeactivateCollections(Request):
    names: Annotated[
        tuple[str, ...],
        Help("the collections to leave out of searches, and of index and embed unless named"),
        Bounds(1, MOST_NAMED),
        NAME_LENGTH,
    ]


@dataclass(frozen=True, slots=True)
class CollectionsSwitched:
    active: bool
    changed: tuple[str, ...]
    already: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ShowStatus(Request):
    names: Annotated[tuple[str, ...], Help("the collections to show (default: all)"), LISTED, NAME_LENGTH] = ()
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
    embedded: int = 0
    model: str | None = None
    fields: tuple[str, ...] = ()
    state: Literal["counted"] = "counted"


@dataclass(frozen=True, slots=True)
class Unavailable(Notice):
    state: Literal["unavailable"] = "unavailable"


@dataclass(frozen=True, slots=True)
class CollectionStatus:
    name: str
    folder: str
    store: str
    contents: Counted | Unavailable
    active: bool = True
    agents: bool = True


@dataclass(frozen=True, slots=True)
class Status:
    home: str
    collections: tuple[CollectionStatus, ...]


@dataclass(frozen=True, slots=True)
class Retitled:
    documents: int
    to_embed: int


NONE_RETITLED = Retitled(0, 0)
