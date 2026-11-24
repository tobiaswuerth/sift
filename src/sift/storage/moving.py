import errno
import os
import sqlite3
import sys
from collections.abc import Callable, Generator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from sift.api.errors import Problem, StoreUnavailableError
from sift.storage.database import primary_code, private_file, private_folder, uri
from sift.storage.unusable import Unusable, unavailable_if_unusable

COPY_PAGES = 16_384
LINKLESS = frozenset({errno.EXDEV, errno.EPERM, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EMLINK})
BUSY_CODES = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})
OPEN_ELSEWHERE = "another sift has the store {path} open: a search, sift index or sift embed"
OPEN_ELSEWHERE_REMEDY = "wait until it is done, or stop that process"
COPY = Unusable("the copy", "check the drive it goes to, then move the collection again")

type Copied = Callable[[int, int], None]


def opened_folder(folder: Path) -> int | None:
    return None if sys.platform == "win32" else os.open(folder, os.O_RDONLY)


@dataclass(frozen=True, slots=True)
class Transfer:
    link: Callable[[Path, Path], None] = os.link
    pages: int = COPY_PAGES
    copy_opened: Callable[[Path], sqlite3.Connection] = sqlite3.connect
    opened_folder: Callable[[Path], int | None] = opened_folder


TRANSFER = Transfer()


def _alone(connection: sqlite3.Connection, path: Path) -> None:
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    try:
        connection.execute("BEGIN EXCLUSIVE")
    except sqlite3.OperationalError as error:
        if primary_code(error) not in BUSY_CODES:
            raise
        said = OPEN_ELSEWHERE.format(path=path)
        raise StoreUnavailableError(said, OPEN_ELSEWHERE_REMEDY, code=Problem.BUSY) from None
    connection.execute("COMMIT")
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")


@contextmanager
def held_alone(path: Path) -> Generator[sqlite3.Connection]:
    with unavailable_if_unusable(path):
        connection = sqlite3.connect(uri(path.resolve(), "rw"), uri=True, timeout=0, isolation_level=None)
    with closing(connection):
        with unavailable_if_unusable(path):
            _alone(connection, path)
        yield connection


def _copied(held: sqlite3.Connection, new: Path, copied: Copied, how: Transfer) -> None:
    part = new.with_name(f"{new.name}.part")
    part.unlink(missing_ok=True)
    private_file(part)
    (size,) = cast("tuple[int]", held.execute("PRAGMA page_size").fetchone())
    (count,) = cast("tuple[int]", held.execute("PRAGMA page_count").fetchone())
    copied(0, count * size)
    try:
        with unavailable_if_unusable(new, COPY), closing(how.copy_opened(part)) as copy:
            held.backup(
                copy, pages=how.pages, progress=lambda _status, left, total: copied((total - left) * size, total * size)
            )
        part.replace(new)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def _synced(folder: Path, opened: Callable[[Path], int | None]) -> None:
    descriptor = opened(folder)
    if descriptor is None:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _linked(held: sqlite3.Connection, old: Path, new: Path, copied: Copied, how: Transfer) -> bool:
    try:
        how.link(old, new)
    except OSError as error:
        if error.errno not in LINKLESS:
            raise
        _copied(held, new, copied, how)
        return False
    return True


def transferred(held: sqlite3.Connection, old: Path, new: Path, copied: Copied, how: Transfer = TRANSFER) -> bool:
    private_folder(new.parent)
    linked = _linked(held, old, new, copied, how)
    _synced(new.parent, how.opened_folder)
    return not linked
