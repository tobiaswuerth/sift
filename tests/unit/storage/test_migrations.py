import contextlib
import sqlite3
from pathlib import Path
from typing import NamedTuple

import pytest

from sift.api.errors import StoreUnavailableError
from sift.storage.database import connect, reading, rows, run, writing
from sift.storage.migrations import Migration, migrated, version

FIRST: Migration = ("CREATE TABLE a(x INTEGER) STRICT",)
SECOND: Migration = ("CREATE TABLE b(y INTEGER) STRICT", "INSERT INTO b(y) VALUES (1)")


class Table(NamedTuple):
    name: str


def tables(path: Path) -> tuple[int, list[str]]:
    with contextlib.closing(connect(path, create=False)) as connection, reading(connection) as transaction:
        found = rows(transaction, Table, "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        return version(transaction), [table.name for table in found]


def migrate(path: Path, migrations: tuple[Migration, ...]) -> int:
    with contextlib.closing(connect(path, create=True)) as connection, writing(connection) as transaction:
        return migrated(transaction, migrations, path)


def test_a_new_database_gets_every_migration_in_order(tmp_path: Path) -> None:
    assert migrate(tmp_path / "a.sqlite", (FIRST, SECOND)) == 0
    assert tables(tmp_path / "a.sqlite") == (2, ["a", "b"])


def test_a_database_gets_the_migrations_it_lacks_alone(tmp_path: Path) -> None:
    migrate(tmp_path / "a.sqlite", (FIRST,))
    assert migrate(tmp_path / "a.sqlite", (FIRST, SECOND)) == 1
    assert tables(tmp_path / "a.sqlite") == (2, ["a", "b"])
    assert migrate(tmp_path / "a.sqlite", (FIRST, SECOND)) == 2


def test_a_database_of_a_newer_sift_is_refused_and_left_as_it_is(tmp_path: Path) -> None:
    migrate(tmp_path / "a.sqlite", (FIRST, SECOND))
    with pytest.raises(StoreUnavailableError, match="newer sift") as refused:
        migrate(tmp_path / "a.sqlite", (FIRST,))
    assert refused.value.remedy == "upgrade sift"
    assert tables(tmp_path / "a.sqlite") == (2, ["a", "b"])


def test_a_migration_that_fails_leaves_the_database_as_it_was(tmp_path: Path) -> None:
    migrate(tmp_path / "a.sqlite", (FIRST,))
    broken: Migration = ("CREATE TABLE c(z INTEGER) STRICT", "INSERT INTO nowhere VALUES (1)")
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        migrate(tmp_path / "a.sqlite", (FIRST, broken))
    assert tables(tmp_path / "a.sqlite") == (1, ["a"])


def test_version_of_a_database_with_none_is_zero(tmp_path: Path) -> None:
    with contextlib.closing(connect(tmp_path / "a.sqlite", create=True)) as connection:
        with writing(connection) as transaction:
            run(transaction, "CREATE TABLE a(x INTEGER) STRICT")
        with reading(connection) as transaction:
            assert version(transaction) == 0
