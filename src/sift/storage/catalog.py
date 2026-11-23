import json
import os
import re
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, cast

from sift.api.collections import TitleFrom
from sift.api.errors import InvalidRequestError, NotFoundError, StoreUnavailableError
from sift.api.ids import CollectionId
from sift.domain.text.records import NO_CHOICE, Chosen
from sift.storage.database import Reading, Writing, connect, inserted, reading, rows, writing
from sift.storage.migrations import Migration, migrated, version
from sift.storage.store import create_store, delete_store

CATALOG = "catalog.sqlite"
STORES = "collections"
STORE_SUFFIX = ".sqlite"
NAME = re.compile(r"\w[\w.-]*")
LAYOUT: tuple[Migration, ...] = (
    (
        """CREATE TABLE collections(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            folder TEXT NOT NULL,
            store TEXT NOT NULL UNIQUE,
            active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
            choices TEXT NOT NULL CHECK (json_valid(choices))
        ) STRICT""",
    ),
)


@dataclass(frozen=True, slots=True)
class Choices:
    title_from: TitleFrom = TitleFrom.HEADING
    follow_links: bool = False
    records: Chosen = NO_CHOICE


DEFAULT_CHOICES = Choices()


@dataclass(frozen=True, slots=True)
class Collection:
    id: CollectionId
    name: str
    folder: Path
    store: Path
    active: bool
    choices: Choices


class Row(NamedTuple):
    id: int
    name: str
    folder: str
    store: str
    active: int
    choices: str


def choices_json(choices: Choices) -> str:
    records = choices.records
    chosen = {"id_field": records.id, "title_field": records.title, "text_fields": list(records.text)}
    return json.dumps({"title_from": choices.title_from.value, "follow_links": choices.follow_links, **chosen})


def _named(value: object, key: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{key} is no text")


def _names(value: object) -> tuple[str, ...]:
    held = cast("list[object]", value) if isinstance(value, list) else None
    if held is None or not all(isinstance(each, str) for each in held):
        raise TypeError("text_fields is no list of texts")
    return tuple(each for each in held if isinstance(each, str))


def _flag(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("follow_links is neither true nor false")
    return value


def choices_of(written: str) -> Choices:
    read = cast("object", json.loads(written))
    if not isinstance(read, dict):
        raise TypeError("they are no mapping")
    document = cast("dict[str, object]", read)
    records = Chosen(
        _named(document.get("id_field"), "id_field"),
        _named(document.get("title_field"), "title_field"),
        _names(document.get("text_fields", [])),
    )
    return Choices(
        title_from=TitleFrom(document.get("title_from", TitleFrom.HEADING.value)),
        follow_links=_flag(document.get("follow_links", False)),
        records=records,
    )


def _collection(row: Row) -> Collection:
    try:
        choices = choices_of(row.choices)
    except (ValueError, TypeError) as error:
        said = f"the catalog's choices for {row.name} were written by a newer sift or damaged: {error}"
        raise StoreUnavailableError(said, "upgrade sift") from None
    return Collection(CollectionId(row.id), row.name, Path(row.folder), Path(row.store), bool(row.active), choices)


def _up_to_date(connection: sqlite3.Connection, path: Path) -> None:
    with reading(connection) as transaction:
        found = version(transaction)
    if found != len(LAYOUT):
        with writing(connection) as transaction:
            migrated(transaction, LAYOUT, path)


def open_catalog(home: Path) -> sqlite3.Connection:
    path = home / CATALOG
    connection = connect(path, create=True)
    try:
        _up_to_date(connection, path)
    except BaseException:
        connection.close()
        raise
    return connection


def collections(transaction: Reading) -> tuple[Collection, ...]:
    found = rows(transaction, Row, "SELECT id, name, folder, store, active, choices FROM collections")
    return tuple(sorted(map(_collection, found), key=lambda known: (known.name.casefold(), known.name)))


def named(known: Sequence[Collection], name: str) -> Collection:
    if found := next((one for one in known if one.name.casefold() == name.casefold()), None):
        return found
    there = ", ".join(one.name for one in known)
    remedy = f"the collections are: {there}" if there else "add a folder as a collection first"
    raise NotFoundError(f"no collection is named {name!r}", remedy)


def collection(transaction: Reading, name: str) -> Collection:
    return named(collections(transaction), name)


def known_collections(home: Path) -> tuple[Collection, ...]:
    if not (home / CATALOG).is_file():
        return ()
    with closing(open_catalog(home)) as connection, reading(connection) as transaction:
        return collections(transaction)


def store_path(home: Path, name: str, given: Path | None) -> Path:
    chosen = (given or home / STORES).resolve()
    return chosen if chosen.name.lower().endswith(STORE_SUFFIX) else chosen / f"{name}{STORE_SUFFIX}"


def _same(folder: Path) -> str:
    return os.path.normcase(folder)


@dataclass(frozen=True, slots=True)
class NewCollection:
    folder: Path
    name: str | None = None
    store: Path | None = None
    nested: bool = False
    choices: Choices = DEFAULT_CHOICES


@dataclass(frozen=True, slots=True)
class Placed:
    folder: Path
    name: str
    store: Path
    nested: bool


def _clash(wanted: Placed, other: Collection) -> InvalidRequestError | None:
    mine, theirs = Path(_same(wanted.folder)), Path(_same(other.folder))
    if mine == theirs:
        return InvalidRequestError(f"{wanted.folder} is the collection {other.name!r} already")
    if other.name.casefold() == wanted.name.casefold():
        return InvalidRequestError(f"a collection is named {other.name!r} already", "choose another name")
    if _same(other.store) == _same(wanted.store):
        return InvalidRequestError(f"{wanted.store} is the store of the collection {other.name!r}")
    if not wanted.nested and (mine.is_relative_to(theirs) or theirs.is_relative_to(mine)):
        where = "lies inside" if mine.is_relative_to(theirs) else "holds"
        return InvalidRequestError(
            f"{wanted.folder} {where} the folder of the collection {other.name!r}: its files would be found twice",
            "add it as nested, if both should hold them",
        )
    return None


def refusal(wanted: Placed, known: tuple[Collection, ...]) -> InvalidRequestError | None:
    if not NAME.fullmatch(wanted.name):
        remedy = "name it with letters, digits, '_', '-' and '.', starting with a letter, a digit or '_'"
        return InvalidRequestError(f"{wanted.name!r} is no collection name", remedy)
    if clash := next(filter(None, (_clash(wanted, other) for other in known)), None):
        return clash
    if wanted.store.exists():
        return InvalidRequestError(f"{wanted.store} exists already", "move it away, or choose another place for it")
    return None


def placed(home: Path, new: NewCollection) -> Placed:
    folder = new.folder.resolve()
    if not folder.is_dir():
        raise InvalidRequestError(f"{folder} is no folder", "give the folder whose files the collection holds")
    name = new.name or folder.name
    return Placed(folder, name, store_path(home, name, new.store), new.nested)


def _registered(transaction: Writing, wanted: Placed, choices: Choices) -> CollectionId:
    if refused := refusal(wanted, collections(transaction)):
        raise refused
    return CollectionId(
        inserted(
            transaction,
            "INSERT INTO collections(name, folder, store, choices) VALUES (?, ?, ?, ?)",
            (wanted.name, str(wanted.folder), str(wanted.store), choices_json(choices)),
        )
    )


def add_collection(connection: sqlite3.Connection, home: Path, new: NewCollection) -> Collection:
    wanted = placed(home, new)
    created = False
    try:
        with writing(connection) as transaction:
            identity = _registered(transaction, wanted, new.choices)
            create_store(wanted.store, identity)
            created = True
    except BaseException:
        if created:
            delete_store(wanted.store)
        raise
    return Collection(identity, wanted.name, wanted.folder, wanted.store, active=True, choices=new.choices)
