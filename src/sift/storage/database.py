import os
import sqlite3
import stat
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import NamedTuple, cast
from urllib.parse import quote

from sift.api.errors import Problem, StoreUnavailableError
from sift.domain.plan.segments import POLICY, Policy

BUSY_TIMEOUT_SECONDS = 30
WAL_LIMIT_BYTES = 64 << 20
CACHE_KIBIBYTES = 256 << 10
DAMAGED = frozenset({sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB})
MINIMUM_SQLITE = (3, 38, 0)
STREAMED = 10_000
SETTINGS = (
    "PRAGMA journal_mode = WAL",
    f"PRAGMA journal_size_limit = {WAL_LIMIT_BYTES}",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA foreign_keys = ON",
    f"PRAGMA cache_size = -{CACHE_KIBIBYTES}",
)

type Parameters = Sequence[object] | Mapping[str, object]
type Connection = sqlite3.Connection
PRIVATE_FOLDER, PRIVATE_FILE, SHARED = 0o700, 0o600, 0o077


@dataclass(frozen=True, slots=True)
class Reading:
    connection: sqlite3.Connection


@dataclass(frozen=True, slots=True)
class Writing(Reading):
    pass


@dataclass(frozen=True, slots=True)
class Batches:
    chunks_a_segment: int = 50_000
    tokens_a_segment: int = 10_000_000
    documents_a_forget: int = 1_000
    postings_a_merge: int = 2_000_000
    bytes_a_merge: int = 64 << 20
    chunks_a_settle: int = 100_000
    merging: Policy = POLICY


BATCHES = Batches()


@dataclass(frozen=True, slots=True)
class WritingDocuments(Writing):
    batches: Batches = BATCHES


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
            code=Problem.SQLITE_TOO_OLD,
        )


def kept_private(path: Path) -> None:
    path.chmod(stat.S_IMODE(path.stat().st_mode) & ~SHARED)


def private_folder(folder: Path) -> None:
    folder.mkdir(mode=PRIVATE_FOLDER, parents=True, exist_ok=True)


def private_file(path: Path) -> None:
    os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, PRIVATE_FILE))


def connect(path: Path, *, create: bool, timeout: float = BUSY_TIMEOUT_SECONDS) -> sqlite3.Connection:
    require_sqlite(sqlite3.sqlite_version_info)
    if create:
        private_folder(path.parent)
        os.close(os.open(path, os.O_CREAT | os.O_WRONLY, PRIVATE_FILE))
    elif not path.is_file():
        raise FileNotFoundError(path)
    kept_private(path)
    connection = sqlite3.connect(
        uri(path.resolve(), "rwc" if create else "rw"),
        uri=True,
        timeout=timeout,
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


def abandon(transaction: Reading) -> None:
    if transaction.connection.in_transaction:
        transaction.connection.execute("ROLLBACK")


@contextmanager
def reading(connection: sqlite3.Connection) -> Generator[Reading]:
    connection.execute("BEGIN")
    held = Reading(connection)
    try:
        yield held
    except BaseException:
        abandon(held)
        raise
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


class Count(NamedTuple):
    n: int


class Value(NamedTuple):
    value: str


class IdRow(NamedTuple):
    id: int


class KeyRow(NamedTuple):
    key: str


def rows[R: NamedTuple](transaction: Reading, shape: type[R], sql: str, parameters: Parameters = ()) -> list[R]:
    fetched = cast("list[tuple[object, ...]]", transaction.connection.execute(sql, parameters).fetchall())
    return [shape._make(row) for row in fetched]


def batches[R: NamedTuple](
    transaction: Reading, shape: type[R], sql: str, parameters: Parameters = (), size: int = STREAMED
) -> Iterator[list[R]]:
    cursor = transaction.connection.execute(sql, parameters)
    while fetched := cast("list[tuple[object, ...]]", cursor.fetchmany(size)):
        yield [shape._make(row) for row in fetched]


def streamed[R: NamedTuple](
    transaction: Reading, shape: type[R], sql: str, parameters: Parameters = (), size: int = STREAMED
) -> Iterator[R]:
    for batch in batches(transaction, shape, sql, parameters, size):
        yield from batch


def first[R: NamedTuple](transaction: Reading, shape: type[R], sql: str, parameters: Parameters = ()) -> R | None:
    fetched = cast("tuple[object, ...] | None", transaction.connection.execute(sql, parameters).fetchone())
    return None if fetched is None else shape._make(fetched)


def run(transaction: Writing, sql: str, parameters: Parameters = ()) -> int:
    return transaction.connection.execute(sql, parameters).rowcount


def run_each(transaction: Writing, sql: str, each: Iterable[Parameters]) -> None:
    transaction.connection.executemany(sql, each)


def save_point(transaction: Writing) -> None:
    run(transaction, "SAVEPOINT written")


def back_to_save_point(transaction: Writing) -> None:
    run(transaction, "ROLLBACK TO written")
    run(transaction, "RELEASE written")


def past_save_point(transaction: Writing) -> None:
    run(transaction, "RELEASE written")


def inserted(transaction: Writing, sql: str, parameters: Parameters = ()) -> int:
    cursor = transaction.connection.execute(sql, parameters)
    if cursor.rowcount != 1 or cursor.lastrowid is None:
        raise ValueError(f"not one row was inserted by: {sql}")
    return cursor.lastrowid
