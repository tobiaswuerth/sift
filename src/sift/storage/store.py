import sqlite3
from collections.abc import Generator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from typing import cast

from sift.api.errors import Problem, StoreUnavailableError
from sift.api.ids import CollectionId, first_row
from sift.domain.codec.shape import SHAPE, Shape
from sift.storage.database import (
    Reading,
    Value,
    connect,
    first,
    private_file,
    private_folder,
    read_only,
    reading,
    run_each,
    writing,
)
from sift.storage.keyword_meta import keyword_meta
from sift.storage.layout import LAYOUT
from sift.storage.migrations import Migration, migrated, newer, version
from sift.storage.store_files import delete_store
from sift.storage.unusable import DAMAGED_REMEDY, raise_for, unavailable_if_unusable

NUMBERED_TABLES = ("documents", "chunks")
MISSING_REMEDY = "connect the drive it lies on, or remove the collection"
LEFT_BY_ADD = """SELECT (SELECT value FROM meta WHERE key = 'collection') = ?
                        AND NOT EXISTS (SELECT 1 FROM documents)"""


SCHEMA_HELD = "SELECT count(*) FROM sqlite_schema"


def holds_nothing(path: Path) -> bool:
    try:
        with closing(read_only(path)) as connection:
            (held,) = cast("tuple[int]", connection.execute(SCHEMA_HELD).fetchone())
    except sqlite3.DatabaseError:
        return False
    return held == 0


def left_by_add(path: Path, collection: CollectionId) -> bool:
    try:
        with closing(read_only(path)) as connection:
            (left,) = cast("tuple[int | None]", connection.execute(LEFT_BY_ADD, (str(collection),)).fetchone())
    except sqlite3.DatabaseError:
        return False
    return left == 1


def _made(path: Path, collection: CollectionId, layout: Sequence[Migration], shape: Shape) -> None:
    connection = connect(path, create=False)
    try:
        with writing(connection) as transaction:
            migrated(transaction, layout, path)
            start = first_row(collection)
            run_each(
                transaction,
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                [("collection", str(collection)), *keyword_meta(transaction, shape, start)],
            )
            run_each(
                transaction,
                "INSERT INTO sqlite_sequence(name, seq) VALUES (?, ?)",
                [(table, start) for table in NUMBERED_TABLES],
            )
    finally:
        connection.close()


def create_store(
    path: Path, collection: CollectionId, layout: Sequence[Migration] = LAYOUT, shape: Shape = SHAPE
) -> None:
    private_folder(path.parent)
    private_file(path)
    try:
        _made(path, collection, layout, shape)
    except BaseException:
        delete_store(path)
        raise


def _checked(connection: sqlite3.Connection, path: Path, collection: CollectionId, layout: Sequence[Migration]) -> None:
    with reading(connection) as transaction:
        found = version(transaction)
        if found == 0:
            raise StoreUnavailableError(f"{path} is no store of sift", DAMAGED_REMEDY, code=Problem.STORE_DAMAGED)
        if found > len(layout):
            raise newer(path, found, len(layout))
        held = first(transaction, Value, "SELECT value FROM meta WHERE key = 'collection'")
    if held is None or held.value != str(collection):
        owner = "no collection" if held is None else f"collection {held.value}"
        said = f"{path} is the store of {owner}, not of collection {collection}"
        raise StoreUnavailableError(said, DAMAGED_REMEDY, code=Problem.STORE_DAMAGED)
    if found < len(layout):
        with writing(connection) as transaction:
            migrated(transaction, layout, path)


def open_store(path: Path, collection: CollectionId, layout: Sequence[Migration] = LAYOUT) -> sqlite3.Connection:
    try:
        connection = connect(path, create=False)
    except FileNotFoundError:
        raise StoreUnavailableError(
            f"the store is missing: {path}", MISSING_REMEDY, code=Problem.STORE_MISSING
        ) from None
    except sqlite3.DatabaseError as error:
        raise_for(path, error)
    try:
        _checked(connection, path, collection, layout)
    except BaseException as error:
        connection.close()
        if isinstance(error, sqlite3.DatabaseError):
            raise_for(path, error)
        raise
    return connection


@contextmanager
def connected(path: Path, collection: CollectionId) -> Generator[sqlite3.Connection]:
    with unavailable_if_unusable(path), closing(open_store(path, collection)) as connection:
        yield connection


@contextmanager
def read_store(path: Path, collection: CollectionId) -> Generator[Reading]:
    with connected(path, collection) as connection, reading(connection) as transaction:
        yield transaction
