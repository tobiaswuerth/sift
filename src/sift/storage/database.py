import sqlite3
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import NamedTuple, cast
from urllib.parse import quote

from sift.api.errors import StoreUnavailableError

BUSY_TIMEOUT_SECONDS = 30
WAL_LIMIT_BYTES = 64 << 20
CACHE_KIBIBYTES = 256 << 10
DAMAGED = frozenset({sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB})
MINIMUM_SQLITE = (3, 37, 0)
SETTINGS = (
    "PRAGMA journal_mode = WAL",
    f"PRAGMA journal_size_limit = {WAL_LIMIT_BYTES}",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA foreign_keys = ON",
    f"PRAGMA cache_size = -{CACHE_KIBIBYTES}",
)

type Parameters = Sequence[object] | Mapping[str, object]
type Connection = sqlite3.Connection


@dataclass(frozen=True, slots=True)
class Reading:
    connection: sqlite3.Connection


@dataclass(frozen=True, slots=True)
class Writing(Reading):
    pass


def uri(path: PurePath, mode: str) -> str:
    written = path.as_posix()
    drive, rest = (written[:2], written[2:]) if written[1:2] == ":" else ("", written)
    return f"file:{'///' if drive else '//'}{drive}{quote(rest)}?mode={mode}"


def primary_code(error: sqlite3.Error) -> int:
    code: object = getattr(error, "sqlite_errorcode", None)
    return code & 0xFF if isinstance(code, int) else 0


def require_sqlite(version: tuple[int, int, int]) -> None:
    if version < MINIMUM_SQLITE:
        found, needed = (".".join(map(str, numbers)) for numbers in (version, MINIMUM_SQLITE))
        raise StoreUnavailableError(
            f"this Python was built with SQLite {found}, which lacks what sift's stores need",
            f"run sift on a Python with SQLite {needed} or newer, as uv, python.org and Homebrew install",
        )


def connect(path: Path, *, create: bool) -> sqlite3.Connection:
    require_sqlite(sqlite3.sqlite_version_info)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(
        uri(path.resolve(), "rwc" if create else "rw"),
        uri=True,
        timeout=BUSY_TIMEOUT_SECONDS,
        isolation_level=None,
        check_same_thread=False,
    )
    try:
        for setting in SETTINGS:
            connection.execute(setting)
    except BaseException:
        connection.close()
        raise
    return connection


@contextmanager
def reading(connection: sqlite3.Connection) -> Generator[Reading]:
    connection.execute("BEGIN")
    try:
        yield Reading(connection)
    finally:
        if connection.in_transaction:
            connection.execute("COMMIT")


@contextmanager
def writing(connection: sqlite3.Connection) -> Generator[Writing]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield Writing(connection)
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def rows[R: NamedTuple](transaction: Reading, shape: type[R], sql: str, parameters: Parameters = ()) -> list[R]:
    fetched = cast("list[tuple[object, ...]]", transaction.connection.execute(sql, parameters).fetchall())
    return [shape._make(row) for row in fetched]


def first[R: NamedTuple](transaction: Reading, shape: type[R], sql: str, parameters: Parameters = ()) -> R | None:
    fetched = cast("tuple[object, ...] | None", transaction.connection.execute(sql, parameters).fetchone())
    return None if fetched is None else shape._make(fetched)


def run(transaction: Writing, sql: str, parameters: Parameters = ()) -> int:
    return transaction.connection.execute(sql, parameters).rowcount


def run_each(transaction: Writing, sql: str, each: Iterable[Parameters]) -> None:
    transaction.connection.executemany(sql, each)


def inserted(transaction: Writing, sql: str, parameters: Parameters = ()) -> int:
    cursor = transaction.connection.execute(sql, parameters)
    if cursor.rowcount != 1 or cursor.lastrowid is None:
        raise ValueError(f"not one row was inserted by: {sql}")
    return cursor.lastrowid
