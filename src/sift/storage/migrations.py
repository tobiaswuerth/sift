import functools
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from sift.api.errors import Problem, StoreUnavailableError
from sift.domain.errors import DamagedError
from sift.storage.database import Count, Reading, Writing, first, rows, run, writing

type Step = str | Callable[[Writing], None]

UNREACHED = (1 << 31) - 1
MARK_MADE = """CREATE TABLE IF NOT EXISTS compatibility(
                   id INTEGER PRIMARY KEY CHECK (id = 1),
                   read_from INTEGER NOT NULL CHECK (read_from >= 1),
                   written_from INTEGER NOT NULL CHECK (written_from >= read_from)
               ) STRICT"""
MARK_SET = "INSERT OR REPLACE INTO compatibility(id, read_from, written_from) VALUES (1, ?, ?)"
MARK_HELD = "SELECT count(*) FROM sqlite_schema WHERE type = 'table' AND name = 'compatibility'"
MARK = "SELECT read_from, written_from FROM compatibility"
READ_ALONE = "this one can read it, not change it"
SCHEMA = "SELECT type, name FROM sqlite_schema WHERE name NOT LIKE 'sqlite!_%' ESCAPE '!'"


class Older(Enum):
    LOCKED_OUT = "locked out"
    READS = "reads"
    WRITES = "writes"


@dataclass(frozen=True, slots=True)
class Migration:
    steps: tuple[Step, ...]
    older: Older = Older.LOCKED_OUT


class Version(NamedTuple):
    number: int


class SchemaObject(NamedTuple):
    type: str
    name: str


class _Marked(NamedTuple):
    read_from: int
    written_from: int


def version(transaction: Reading) -> int:
    (found,) = rows(transaction, Version, "PRAGMA user_version")
    return found.number


def newer(path: Path, found: int, known: int) -> StoreUnavailableError:
    return StoreUnavailableError(
        f"{path} was written by a newer sift (layout {found}; this sift knows up to {known})",
        "upgrade sift",
        code=Problem.STORE_NEWER,
    )


def _read_only(path: Path, found: int, known: int) -> StoreUnavailableError:
    said = f"{path} was changed by a newer sift (layout {found}; this sift knows up to {known})"
    return StoreUnavailableError(f"{said}: {READ_ALONE}", "upgrade sift", code=Problem.STORE_NEWER)


def _marked(transaction: Reading) -> _Marked:
    (held,) = rows(transaction, Count, MARK_HELD)
    found = first(transaction, _Marked, MARK) if held.n else None
    return _Marked(UNREACHED, UNREACHED) if found is None else found


def readable_version(transaction: Reading, known: int, path: Path) -> int:
    found = version(transaction)
    if found > known and known < _marked(transaction).read_from:
        raise newer(path, found, known)
    return found


def writable_version(transaction: Reading, known: int, path: Path) -> int:
    found = readable_version(transaction, known, path)
    if found > known and known < _marked(transaction).written_from:
        raise _read_only(path, found, known)
    return found


def _taken(transaction: Writing, step: Step) -> None:
    if isinstance(step, str):
        run(transaction, step)
    else:
        step(transaction)


@functools.cache
def _made(migrations: tuple[Migration, ...]) -> frozenset[SchemaObject]:
    with closing(sqlite3.connect(":memory:", isolation_level=None)) as made, writing(made) as transaction:
        for migration in migrations:
            for step in migration.steps:
                _taken(transaction, step)
        run(transaction, MARK_MADE)
        return frozenset(rows(transaction, SchemaObject, SCHEMA))


def refuse_lacking(transaction: Reading, migrations: Sequence[Migration], path: Path) -> None:
    found, held = version(transaction), set(rows(transaction, SchemaObject, SCHEMA))
    lacked = sorted(_made(tuple(migrations[:found])) - held, key=lambda each: (each.type != "table", each))
    if lacked and found > len(migrations):
        raise newer(path, found, len(migrations))
    if lacked:
        kind, name = lacked[0]
        raise DamagedError(f"it lacks the {kind} {name}")


def _last(migrations: Sequence[Migration], needed: frozenset[Older]) -> int:
    return max((number for number, each in enumerate(migrations, 1) if each.older in needed), default=1)


def migrated(transaction: Writing, migrations: Sequence[Migration], path: Path) -> int:
    start = version(transaction)
    if start > len(migrations):
        raise newer(path, start, len(migrations))
    if start == len(migrations):
        return start
    for migration in migrations[start:]:
        for step in migration.steps:
            _taken(transaction, step)
    read_from = _last(migrations, frozenset({Older.LOCKED_OUT}))
    written_from = _last(migrations, frozenset({Older.LOCKED_OUT, Older.READS}))
    run(transaction, MARK_MADE)
    run(transaction, MARK_SET, (read_from, written_from))
    run(transaction, f"PRAGMA user_version = {len(migrations)}")
    return start
