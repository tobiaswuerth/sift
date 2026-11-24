import json
import os
import re
import sqlite3
from collections.abc import Generator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, cast

from sift.api.collections import TitleFrom
from sift.api.errors import InvalidRequestError, NotFoundError, Problem, StoreUnavailableError
from sift.api.ids import CollectionId
from sift.domain.text.records import NO_CHOICE, Chosen
from sift.storage.database import (
    BUSY_TIMEOUT_SECONDS,
    Reading,
    Writing,
    connect,
    inserted,
    kept_private,
    read_only,
    reading,
    rows,
    run,
    run_each,
    writing,
)
from sift.storage.migrations import Migration, migrated, version
from sift.storage.store import create_store, holds_nothing, left_by_add
from sift.storage.store_files import delete_store
from sift.storage.unusable import Unusable, unavailable_if_unusable

CATALOG = "catalog.sqlite"
CATALOG_DAMAGED_REMEDY = "move the catalog away, then add the folders again"
THE_CATALOG = Unusable("the catalog", CATALOG_DAMAGED_REMEDY)
WRITTEN_THROUGH = "PRAGMA synchronous = FULL"
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
STORE_TAKEN_REMEDY = "move it away, or choose another place for it"
INACTIVE_REMEDY = "run sift activate with its name"


@dataclass(frozen=True, slots=True)
class Choices:
    title_from: TitleFrom = TitleFrom.HEADING
    follow_links: bool = False
    records: Chosen = NO_CHOICE
    agents: bool = True


DEFAULT_CHOICES = Choices()


@dataclass(frozen=True, slots=True)
class Collection:
    id: CollectionId
    name: str
    folder: Path
    store: Path
    active: bool
    choices: Choices


class CollectionRow(NamedTuple):
    id: int
    name: str
    folder: str
    store: str
    active: int
    choices: str


def choices_json(choices: Choices) -> str:
    records = choices.records
    chosen = {"id_field": records.id, "title_field": records.title, "text_fields": list(records.text)}
    flags = {"follow_links": choices.follow_links, "agents": choices.agents}
    return json.dumps({"title_from": choices.title_from.value, **flags, **chosen})


def _named(value: object, key: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{key} is no text")


def _names(value: object) -> tuple[str, ...]:
    held = cast("list[object]", value) if isinstance(value, list) else None
    if held is None or not all(isinstance(each, str) for each in held):
        raise TypeError("text_fields is no list of texts")
    return tuple(each for each in held if isinstance(each, str))


def _flag(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{key} is neither true nor false")
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
        follow_links=_flag(document.get("follow_links", False), "follow_links"),
        records=records,
        agents=_flag(document.get("agents", True), "agents"),
    )


def _collection(row: CollectionRow) -> Collection:
    try:
        choices = choices_of(row.choices)
    except (ValueError, TypeError) as error:
        said = f"the catalog's choices for {row.name} were written by a newer sift or damaged: {error}"
        raise StoreUnavailableError(said, "upgrade sift", code=Problem.STORE_NEWER) from None
    return Collection(CollectionId(row.id), row.name, Path(row.folder), Path(row.store), bool(row.active), choices)


def _up_to_date(connection: sqlite3.Connection, path: Path) -> None:
    with reading(connection) as transaction:
        found = version(transaction)
    if found != len(LAYOUT):
        with writing(connection) as transaction:
            migrated(transaction, LAYOUT, path)


def open_catalog(home: Path, timeout: float = BUSY_TIMEOUT_SECONDS) -> sqlite3.Connection:
    path = home / CATALOG
    connection = connect(path, create=True, timeout=timeout)
    kept_private(home)
    try:
        connection.execute(WRITTEN_THROUGH)
        _up_to_date(connection, path)
    except BaseException:
        connection.close()
        raise
    return connection


@contextmanager
def catalog_of(home: Path, timeout: float = BUSY_TIMEOUT_SECONDS) -> Generator[sqlite3.Connection]:
    with unavailable_if_unusable(home / CATALOG, THE_CATALOG), closing(open_catalog(home, timeout)) as connection:
        yield connection


def collections(transaction: Reading) -> tuple[Collection, ...]:
    found = rows(transaction, CollectionRow, "SELECT id, name, folder, store, active, choices FROM collections")
    return tuple(sorted(map(_collection, found), key=lambda known: (known.name.casefold(), known.name)))


def named(known: Sequence[Collection], name: str) -> Collection:
    if found := next((one for one in known if one.name.casefold() == name.casefold()), None):
        return found
    there = ", ".join(one.name for one in known)
    remedy = f"the collections are: {there}" if there else "add a folder as a collection first"
    raise NotFoundError(f"no collection is named {name!r}", remedy, code=Problem.UNKNOWN_COLLECTION)


def refuse_inactive(one: Collection) -> Collection:
    if not one.active:
        said = f"the collection {one.name!r} is inactive"
        raise InvalidRequestError(said, INACTIVE_REMEDY, code=Problem.INACTIVE_COLLECTION)
    return one


def known_collections(home: Path) -> tuple[Collection, ...]:
    path = home / CATALOG
    if not path.is_file():
        return ()
    with (
        unavailable_if_unusable(path, THE_CATALOG),
        closing(read_only(path)) as connection,
        reading(connection) as transaction,
    ):
        if version(transaction) == len(LAYOUT):
            return collections(transaction)
    with catalog_of(home) as connection, reading(connection) as transaction:
        return collections(transaction)


def forget_collection(home: Path, identity: CollectionId) -> None:
    with catalog_of(home) as connection, writing(connection) as transaction:
        run(transaction, "DELETE FROM collections WHERE id = ?", (identity,))


def set_choices(home: Path, identity: CollectionId, choices: Choices) -> None:
    with catalog_of(home) as connection, writing(connection) as transaction:
        run(transaction, "UPDATE collections SET choices = ? WHERE id = ?", (choices_json(choices), identity))


def holding_folder(home: Path, folder: Path) -> Collection | None:
    wanted = _same(folder.resolve())
    return next((one for one in known_collections(home) if _same(one.folder) == wanted), None)


def set_store(home: Path, identity: CollectionId, store: Path) -> None:
    with catalog_of(home) as connection, writing(connection) as transaction:
        run(transaction, "UPDATE collections SET store = ? WHERE id = ?", (str(store), identity))


def same_place(one: Path, other: Path) -> bool:
    return _same(one.resolve()) == _same(other.resolve())


def all_named(known: Sequence[Collection], names: Sequence[str]) -> tuple[Collection, ...]:
    return tuple(dict.fromkeys(named(known, name) for name in names))


def set_active(home: Path, names: Sequence[str], *, active: bool) -> tuple[Collection, ...]:
    if not (home / CATALOG).is_file():
        return all_named((), names)
    with catalog_of(home) as connection, writing(connection) as transaction:
        chosen = all_named(collections(transaction), names)
        run_each(
            transaction,
            "UPDATE collections SET active = ? WHERE id = ?",
            [(int(active), one.id) for one in chosen],
        )
    return chosen


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
        return InvalidRequestError(f"{wanted.folder} is the collection {other.name!r} already", "index it instead")
    if other.name.casefold() == wanted.name.casefold():
        return InvalidRequestError(f"a collection is named {other.name!r} already", "choose another name")
    if _same(other.store) == _same(wanted.store):
        return InvalidRequestError(
            f"{wanted.store} is the store of the collection {other.name!r}", "give the new collection another name"
        )
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


def _cleared(home: Path, store: Path, identity: CollectionId) -> None:
    if not store.exists():
        return
    made_here = store.parent == home / STORES and holds_nothing(store)
    if not made_here and not left_by_add(store, identity):
        raise InvalidRequestError(f"{store} exists already", STORE_TAKEN_REMEDY)
    delete_store(store)


def add_collection(connection: sqlite3.Connection, home: Path, new: NewCollection) -> Collection:
    wanted = placed(home, new)
    created = False
    try:
        with writing(connection) as transaction:
            identity = _registered(transaction, wanted, new.choices)
            _cleared(home, wanted.store, identity)
            with unavailable_if_unusable(wanted.store):
                create_store(wanted.store, identity)
            created = True
    except BaseException:
        if created:
            delete_store(wanted.store)
        raise
    return Collection(identity, wanted.name, wanted.folder, wanted.store, active=True, choices=new.choices)
