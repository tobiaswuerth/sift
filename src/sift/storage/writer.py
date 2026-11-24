import os
import sqlite3
from collections.abc import Callable, Generator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from sift.api.errors import Problem, StoreUnavailableError
from sift.storage.database import PRIVATE_FILE, primary_code, private_folder, require_sqlite, uri
from sift.storage.store import unavailable_if_unusable
from sift.storage.store_files import WRITER_SUFFIX

BUSY_REMEDY = "wait until it is done, or stop that process"
HOLDER = "CREATE TABLE holder(pid INTEGER NOT NULL, purpose TEXT NOT NULL) STRICT"


class LockFile(Protocol):
    def execute(self, sql: str, parameters: tuple[int | str, ...] = (), /) -> sqlite3.Cursor: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Holder:
    pid: int
    purpose: str


def _made(connection: LockFile) -> None:
    (version,) = cast("tuple[int]", connection.execute("PRAGMA user_version").fetchone())
    if version == 0:
        connection.execute(HOLDER)
        connection.execute("PRAGMA user_version = 1")


def _holder(connection: LockFile) -> Holder | None:
    found = cast("tuple[int, str] | None", connection.execute("SELECT pid, purpose FROM holder").fetchone())
    return None if found is None else Holder(*found)


def _busy(connection: LockFile) -> StoreUnavailableError:
    holder = _holder(connection)
    who = "" if holder is None else f" ({holder.purpose}, process {holder.pid})"
    return StoreUnavailableError(f"another sift writes it now{who}", BUSY_REMEDY, code=Problem.BUSY)


def _hold(connection: LockFile) -> None:
    try:
        connection.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as error:
        if primary_code(error) != sqlite3.SQLITE_BUSY:
            raise
        raise _busy(connection) from None


def _taken(connection: LockFile, purpose: str) -> None:
    _hold(connection)
    connection.execute("DELETE FROM holder")
    connection.execute("INSERT INTO holder(pid, purpose) VALUES (?, ?)", (os.getpid(), purpose))
    connection.execute("COMMIT")
    _hold(connection)


def lock_file(path: Path) -> LockFile:
    return sqlite3.connect(uri(path.resolve(), "rw"), uri=True, timeout=0, isolation_level=None)


def held_by(path: Path, opened: Callable[[Path], LockFile] = lock_file) -> Holder | None:
    if not path.is_file():
        return None
    with closing(opened(path)) as connection, unavailable_if_unusable(path):
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as error:
            if primary_code(error) != sqlite3.SQLITE_BUSY:
                raise
            return _holder(connection)
        connection.execute("ROLLBACK")
        return None


@contextmanager
def sole_writer(store: Path, purpose: str, opened: Callable[[Path], LockFile] = lock_file) -> Generator[None]:
    require_sqlite(sqlite3.sqlite_version_info)
    path = Path(f"{store}{WRITER_SUFFIX}")
    private_folder(path.parent)
    os.close(os.open(path, os.O_CREAT | os.O_RDONLY, PRIVATE_FILE))
    with closing(opened(path)) as connection:
        with unavailable_if_unusable(path):
            _made(connection)
            _taken(connection, purpose)
        try:
            yield
        finally:
            connection.execute("ROLLBACK")
