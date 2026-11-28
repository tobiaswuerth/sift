import os
import sqlite3
from collections.abc import Callable, Generator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from sift.api.errors import Problem, StoreUnavailableError
from sift.storage.database import primary_code, private_file_once, private_folder, require_sqlite, uri
from sift.storage.store_files import WRITER_SUFFIX
from sift.storage.unusable import LOCK, unavailable_if_unusable

BUSY_REMEDY = "wait until it is done, or stop that process"
MOMENT_SECONDS = 0.5
ASK_AT_ONCE = "PRAGMA busy_timeout = 0"
ASK_AGAIN_SECONDS = 0.05
ASK_AGAIN = f"PRAGMA busy_timeout = {round(ASK_AGAIN_SECONDS * 1000)}"
WAIT_A_MOMENT = f"PRAGMA busy_timeout = {round(MOMENT_SECONDS * 1000)}"
HOLDER = "CREATE TABLE IF NOT EXISTS holder(pid INTEGER NOT NULL, purpose TEXT NOT NULL) STRICT"
HELD = "SELECT name FROM sqlite_schema WHERE type = 'table' AND name = 'holder'"
WRITING = "BEGIN IMMEDIATE"
NAMING = "BEGIN EXCLUSIVE"
NAMINGS = 3


class LockFile(Protocol):
    def execute(self, sql: str, parameters: tuple[int | str, ...] = (), /) -> sqlite3.Cursor: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Holder:
    pid: int
    purpose: str


def _holder(connection: LockFile) -> Holder | None:
    try:
        if connection.execute(HELD).fetchone() is None:
            return None
        found = cast("tuple[int, str] | None", connection.execute("SELECT pid, purpose FROM holder").fetchone())
    except sqlite3.OperationalError as error:
        if primary_code(error) != sqlite3.SQLITE_BUSY:
            raise
        return None
    return None if found is None else Holder(*found)


def _began(connection: LockFile, begin: str) -> bool:
    try:
        connection.execute(begin)
    except sqlite3.OperationalError as error:
        if primary_code(error) != sqlite3.SQLITE_BUSY:
            raise
        return False
    return True


def _refused(holder: Holder | None) -> StoreUnavailableError:
    who = "" if holder is None else f" ({holder.purpose}, process {holder.pid})"
    return StoreUnavailableError(f"another sift writes it now{who}", BUSY_REMEDY, code=Problem.BUSY)


def _hold(connection: LockFile, begin: str, own: Holder | None = None) -> None:
    if not _began(connection, begin):
        holder = _holder(connection)
        raise _refused(None if holder == own else holder)


def _named(connection: LockFile, own: Holder) -> None:
    connection.execute(HOLDER)
    connection.execute("PRAGMA user_version = 1")
    connection.execute("DELETE FROM holder")
    connection.execute("INSERT INTO holder(pid, purpose) VALUES (?, ?)", (own.pid, own.purpose))
    connection.execute("COMMIT")


def _taken(connection: LockFile, purpose: str) -> None:
    own = Holder(os.getpid(), purpose)
    named: Holder | None = None
    holder: Holder | None = None
    for _ in range(NAMINGS):
        _hold(connection, NAMING, named)
        _named(connection, own)
        named = own
        _hold(connection, WRITING, named)
        holder = _holder(connection)
        if holder == own:
            return
        connection.execute("ROLLBACK")
    raise _refused(holder)


def lock_file(path: Path) -> LockFile:
    return sqlite3.connect(uri(path.resolve(), "rw"), uri=True, timeout=MOMENT_SECONDS, isolation_level=None)


def _free(connection: LockFile, waiting: str) -> bool:
    connection.execute(waiting)
    if not _began(connection, WRITING):
        return False
    connection.execute("ROLLBACK")
    return True


def held_by(path: Path, opened: Callable[[Path], LockFile] = lock_file) -> Holder | None:
    if not path.is_file():
        return None
    with closing(opened(path)) as connection, unavailable_if_unusable(path, LOCK):
        if _free(connection, ASK_AT_ONCE) or _free(connection, ASK_AGAIN):
            return None
        connection.execute(WAIT_A_MOMENT)
        return _holder(connection)


@contextmanager
def sole_writer(store: Path, purpose: str, opened: Callable[[Path], LockFile] = lock_file) -> Generator[None]:
    require_sqlite(sqlite3.sqlite_version_info)
    path = Path(f"{store}{WRITER_SUFFIX}")
    private_folder(path.parent)
    private_file_once(path)
    with closing(opened(path)) as connection:
        with unavailable_if_unusable(path, LOCK):
            _taken(connection, purpose)
        try:
            yield
        finally:
            connection.execute("ROLLBACK")
