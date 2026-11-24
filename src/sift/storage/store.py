import sqlite3
from collections.abc import Callable, Generator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from typing import cast

from sift.api.errors import Problem, StoreUnavailableError
from sift.api.ids import CollectionId, first_row
from sift.domain.codec.shape import SHAPE, Shape
from sift.storage.database import (
    READ_CACHE,
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
from sift.storage.unusable import DAMAGED_REMEDY, unavailable_if_unusable
from sift.storage.writer import sole_writer

NUMBERED_TABLES = ("documents", "chunks")
MISSING_REMEDY = "connect the drive it lies on, or run sift rm with its name"
BRINGING_UP_TO_DATE = "bringing the store up to date"
LEFT_BY_ADD = """SELECT (SELECT value FROM meta WHERE key = 'collection') = ?
                        AND NOT EXISTS (SELECT 1 FROM documents)"""


SCHEMA_HELD = "SELECT count(*) FROM sqlite_schema"
COLLECTION_HELD = "SELECT value FROM meta WHERE key = 'collection'"


def _probed(path: Path, sql: str, parameters: tuple[str, ...] = ()) -> tuple[object, ...] | None:
    try:
        with closing(read_only(path, immutable=not Path(f"{path}-wal").exists())) as connection:
            return cast("tuple[object, ...] | None", connection.execute(sql, parameters).fetchone())
    except sqlite3.DatabaseError:
        return None


def holds_nothing(path: Path) -> bool:
    return _probed(path, SCHEMA_HELD) == (0,)


def left_by_add(path: Path, collection: CollectionId) -> bool:
    return _probed(path, LEFT_BY_ADD, (str(collection),)) == (1,)


def of_another_collection(path: Path, collection: CollectionId) -> bool:
    held = _probed(path, COLLECTION_HELD)
    return held is not None and held != (str(collection),)


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


def _verified(transaction: Reading, path: Path, collection: CollectionId, layout: Sequence[Migration]) -> int:
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
    return found


def _writable(path: Path) -> sqlite3.Connection:
    return connect(path, create=False)


def _opened(path: Path, opener: Callable[[Path], sqlite3.Connection]) -> sqlite3.Connection:
    if not path.is_file():
        raise StoreUnavailableError(f"the store is missing: {path}", MISSING_REMEDY, code=Problem.STORE_MISSING)
    with unavailable_if_unusable(path):
        return opener(path)


@contextmanager
def _closed_if_failed(connection: sqlite3.Connection) -> Generator[None]:
    try:
        yield
    except BaseException:
        connection.close()
        raise


def open_store(path: Path, collection: CollectionId, layout: Sequence[Migration] = LAYOUT) -> sqlite3.Connection:
    connection = _opened(path, _writable)
    with unavailable_if_unusable(path), _closed_if_failed(connection):
        with reading(connection) as transaction:
            found = _verified(transaction, path, collection, layout)
        if found < len(layout):
            with writing(connection) as transaction:
                migrated(transaction, layout, path)
    return connection


def _read_only_store(
    path: Path, collection: CollectionId, layout: Sequence[Migration]
) -> tuple[sqlite3.Connection, int]:
    connection = _opened(path, read_only)
    with unavailable_if_unusable(path), _closed_if_failed(connection):
        connection.execute(READ_CACHE)
        with reading(connection) as transaction:
            return connection, _verified(transaction, path, collection, layout)


def _for_reading(path: Path, collection: CollectionId, layout: Sequence[Migration] = LAYOUT) -> sqlite3.Connection:
    connection, found = _read_only_store(path, collection, layout)
    if found == len(layout):
        return connection
    connection.close()
    with sole_writer(path, BRINGING_UP_TO_DATE), closing(open_store(path, collection, layout)):
        pass
    connection, _ = _read_only_store(path, collection, layout)
    return connection


@contextmanager
def connected(path: Path, collection: CollectionId) -> Generator[sqlite3.Connection]:
    with unavailable_if_unusable(path), closing(open_store(path, collection)) as connection:
        yield connection


@contextmanager
def read_connected(
    path: Path, collection: CollectionId, layout: Sequence[Migration] = LAYOUT
) -> Generator[sqlite3.Connection]:
    with unavailable_if_unusable(path), closing(_for_reading(path, collection, layout)) as connection:
        yield connection


@contextmanager
def read_store(path: Path, collection: CollectionId, layout: Sequence[Migration] = LAYOUT) -> Generator[Reading]:
    with read_connected(path, collection, layout) as connection, reading(connection) as transaction:
        yield transaction
