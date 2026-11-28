import contextlib
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
READ_CACHE_KIBIBYTES = 16 << 10
DAMAGED = frozenset({sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB})
LEAST_SQLITE = (3, 38, 0)
STREAMED = 10_000
MOST_VARIABLES = 32_766
CACHE = f"PRAGMA cache_size = -{CACHE_KIBIBYTES}"
READ_CACHE = f"PRAGMA cache_size = -{READ_CACHE_KIBIBYTES}"
WAL = "wal"
JOURNAL_MODE = "PRAGMA journal_mode"
NO_LOG = "{path} lies where SQLite keeps no write-ahead log (it keeps its journal by {mode})"
NO_LOG_REMEDY = "keep sift's stores on a drive of this machine, not on a network share"
SETTINGS = (
    "PRAGMA journal_mode = WAL",
    f"PRAGMA journal_size_limit = {WAL_LIMIT_BYTES}",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA foreign_keys = ON",
    CACHE,
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
    chunks_a_read: int = 1_000
    chunks_a_read_beside: int = 250
    readers: int = 8
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
    if version < LEAST_SQLITE:
        found, needed = (".".join(map(str, numbers)) for numbers in (version, LEAST_SQLITE))
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


def private_bytes(path: Path, data: bytes) -> None:
    private_folder(path.parent)
    part = path.with_name(f"{path.name}.{os.getpid()}.part")
    part.unlink(missing_ok=True)
    private_file(part)
    try:
        part.write_bytes(data)
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def private_file_once(path: Path) -> None:
    with contextlib.suppress(FileExistsError):
        private_file(path)


def read_only(path: Path, timeout: float = BUSY_TIMEOUT_SECONDS, *, immutable: bool = False) -> sqlite3.Connection:
    require_sqlite(sqlite3.sqlite_version_info)
    whole = uri(path.resolve(), "ro") + ("&immutable=1" if immutable else "")
    return sqlite3.connect(whole, uri=True, timeout=timeout, isolation_level=None, check_same_thread=False)


def file_of(connection: sqlite3.Connection) -> Path:
    attached = cast("list[tuple[int, str, str]]", connection.execute("PRAGMA database_list").fetchall())
    (main,) = [file for _, name, file in attached if name == "main"]
    return Path(main)


def _refuse_without_a_log(connection: sqlite3.Connection, path: Path) -> None:
    (mode,) = cast("tuple[str]", connection.execute(JOURNAL_MODE).fetchone())
    if mode != WAL:
        raise StoreUnavailableError(NO_LOG.format(path=path, mode=mode), NO_LOG_REMEDY)


def connect(
    path: Path, *, create: bool, timeout: float = BUSY_TIMEOUT_SECONDS, settings: Sequence[str] = SETTINGS
) -> sqlite3.Connection:
    require_sqlite(sqlite3.sqlite_version_info)
    if create:
        private_folder(path.parent)
        private_file_once(path)
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
        for setting in settings:
            connection.execute(setting)
        _refuse_without_a_log(connection, path)
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


def row_batches(
    transaction: Reading, sql: str, parameters: Parameters = (), size: int = STREAMED
) -> Iterator[list[tuple[object, ...]]]:
    cursor = transaction.connection.execute(sql, parameters)
    while fetched := cast("list[tuple[object, ...]]", cursor.fetchmany(size)):
        yield fetched


def batches[R: NamedTuple](
    transaction: Reading, shape: type[R], sql: str, parameters: Parameters = (), size: int = STREAMED
) -> Iterator[list[R]]:
    for fetched in row_batches(transaction, sql, parameters, size):
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


def _values(width: int, count: int) -> str:
    one = f"({', '.join('?' * width)})"
    return ", ".join([one] * count)


def run_rows(transaction: Writing, insert: str, each: Sequence[Sequence[object]]) -> None:
    width = len(each[0]) if each else 1
    most = MOST_VARIABLES // width
    for start in range(0, len(each), most):
        part = each[start : start + most]
        statement = " ".join((insert, "VALUES", _values(width, len(part))))
        transaction.connection.execute(statement, [value for row in part for value in row])


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
