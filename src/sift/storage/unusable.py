import sqlite3
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

from sift.api.errors import Problem, StoreUnavailableError
from sift.domain.errors import DamagedError
from sift.storage.database import DAMAGED, primary_code

DAMAGED_REMEDY = "run sift rm with its name, then add its folder and index it again"
BUSY = ("another sift is writing to {what} {path}", "wait for it to finish, then run sift again")
NOT_WRITABLE = (
    "{what} {path} cannot be written: {error}",
    "see that its drive is connected and writable and that the file is not read-only",
)
UNUSABLE: Mapping[int, tuple[str, str]] = MappingProxyType(
    {
        sqlite3.SQLITE_BUSY: BUSY,
        sqlite3.SQLITE_LOCKED: BUSY,
        sqlite3.SQLITE_FULL: ("the disk holding {what} {path} is full", "free space on it, then run sift again"),
        sqlite3.SQLITE_READONLY: NOT_WRITABLE,
        sqlite3.SQLITE_CANTOPEN: NOT_WRITABLE,
        sqlite3.SQLITE_PERM: NOT_WRITABLE,
        sqlite3.SQLITE_IOERR: ("{what} {path} could not be read or written: {error}", "check the drive it lies on"),
    }
)


@dataclass(frozen=True, slots=True)
class Unusable:
    what: str
    damaged_remedy: str


STORE = Unusable("the store", DAMAGED_REMEDY)
LOCK = Unusable("the lock file", "delete it: sift makes it again")


def damaged(path: Path, error: Exception, unusable: Unusable = STORE) -> StoreUnavailableError:
    said = f"{unusable.what} {path} is damaged: {error}"
    return StoreUnavailableError(said, unusable.damaged_remedy, code=Problem.STORE_DAMAGED)


def raise_for(path: Path, error: sqlite3.DatabaseError, unusable: Unusable = STORE) -> NoReturn:
    code = primary_code(error)
    if code in DAMAGED:
        raise damaged(path, error, unusable) from None
    if code in UNUSABLE:
        said, remedy = UNUSABLE[code]
        raise StoreUnavailableError(said.format(what=unusable.what, path=path, error=error), remedy) from None
    raise error


@contextmanager
def unavailable_if_unusable(path: Path, unusable: Unusable = STORE) -> Generator[None]:
    try:
        yield
    except sqlite3.DatabaseError as error:
        raise_for(path, error, unusable)
    except DamagedError as error:
        raise damaged(path, error, unusable) from None
