import contextlib
import sqlite3
from pathlib import Path, PureWindowsPath
from typing import NamedTuple

import pytest

from sift.api.errors import StoreUnavailableError
from sift.storage.database import (
    MINIMUM_SQLITE,
    WAL_LIMIT_BYTES,
    connect,
    first,
    inserted,
    primary_code,
    reading,
    require_sqlite,
    rows,
    run,
    run_each,
    uri,
    writing,
)
from tests.damage import damage_table


class Setting(NamedTuple):
    value: str | int


class Item(NamedTuple):
    id: int
    name: str


def database(path: Path) -> contextlib.closing[sqlite3.Connection]:
    connection = connect(path, create=True)
    with writing(connection) as transaction:
        run(transaction, "CREATE TABLE items(id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE) STRICT")
    return contextlib.closing(connection)


@pytest.mark.parametrize(
    "pragma, value", [("journal_mode", "wal"), ("foreign_keys", 1), ("synchronous", 1), ("cache_size", -262144)]
)
def test_a_database_is_opened_with_sift_s_settings(tmp_path: Path, pragma: str, value: str | int) -> None:
    with database(tmp_path / "a.sqlite") as connection, reading(connection) as transaction:
        assert first(transaction, Setting, f"PRAGMA {pragma}") == Setting(value)


def test_a_missing_database_is_not_made_unless_asked(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        connect(tmp_path / "gone.sqlite", create=False)
    assert list(tmp_path.iterdir()) == []


def test_a_new_database_gets_its_folders(tmp_path: Path) -> None:
    with database(tmp_path / "a" / "b" / "c.sqlite"):
        assert (tmp_path / "a" / "b" / "c.sqlite").is_file()


@pytest.mark.parametrize(
    "path, expected",
    [
        (PureWindowsPath("D:/data/my notes.sqlite"), "file:///D:/data/my%20notes.sqlite?mode=rw"),
        (PureWindowsPath("//nas/share/a.sqlite"), "file:////nas/share/a.sqlite?mode=rw"),
        (Path("/home/a/b#c.sqlite"), "file:///home/a/b%23c.sqlite?mode=rw"),
    ],
)
def test_a_path_becomes_a_uri_sqlite_takes(path: Path, expected: str) -> None:
    assert uri(path, "rw") == expected


def test_a_write_is_kept_once_its_transaction_ends(tmp_path: Path) -> None:
    with database(tmp_path / "a.sqlite") as connection:
        with writing(connection) as transaction:
            assert inserted(transaction, "INSERT INTO items(name) VALUES ('a')") == 1
            run_each(transaction, "INSERT INTO items(name) VALUES (?)", [("b",), ("c",)])
        with reading(connection) as transaction:
            assert rows(transaction, Item, "SELECT id, name FROM items ORDER BY id") == [
                Item(1, "a"),
                Item(2, "b"),
                Item(3, "c"),
            ]


def insert_twice(connection: sqlite3.Connection) -> None:
    with writing(connection) as transaction:
        run(transaction, "INSERT INTO items(name) VALUES ('a')")
        run(transaction, "INSERT INTO items(name) VALUES ('a')")


def fail_after_sqlite_rolled_back(connection: sqlite3.Connection) -> None:
    with writing(connection) as transaction:
        run(transaction, "INSERT INTO items(name) VALUES ('b')")
        run(transaction, "ROLLBACK")
        raise KeyError


def test_a_transaction_that_fails_leaves_nothing_of_itself(tmp_path: Path) -> None:
    with database(tmp_path / "a.sqlite") as connection:
        with pytest.raises(sqlite3.IntegrityError):
            insert_twice(connection)
        with pytest.raises(KeyError):
            fail_after_sqlite_rolled_back(connection)
        with reading(connection) as transaction:
            assert first(transaction, Item, "SELECT id, name FROM items") is None


def test_a_reading_sees_the_database_as_it_was_when_it_began(tmp_path: Path) -> None:
    with (
        database(tmp_path / "a.sqlite") as connection,
        contextlib.closing(connect(tmp_path / "a.sqlite", create=False)) as other,
    ):
        with reading(connection) as transaction:
            assert rows(transaction, Item, "SELECT id, name FROM items") == []
            with writing(other) as elsewhere:
                run(elsewhere, "INSERT INTO items(name) VALUES ('a')")
            assert rows(transaction, Item, "SELECT id, name FROM items") == []
        with reading(connection) as transaction:
            assert rows(transaction, Item, "SELECT id, name FROM items") == [Item(1, "a")]


def log_size(path: Path) -> int:
    return path.with_name(f"{path.name}-wal").stat().st_size


def test_a_write_ahead_log_grown_while_read_shrinks_back_to_its_limit_once_the_reading_ends(tmp_path: Path) -> None:
    path = tmp_path / "a.sqlite"
    with database(path) as connection, contextlib.closing(connect(path, create=False)) as other:
        with writing(connection) as transaction:
            run(transaction, "CREATE TABLE blobs(data BLOB NOT NULL) STRICT")
        with reading(other) as held:
            assert rows(held, Item, "SELECT id, name FROM items") == []
            for _ in range(24):
                with writing(connection) as transaction:
                    run(transaction, "INSERT INTO blobs(data) VALUES (randomblob(4000000))")
            assert log_size(path) > WAL_LIMIT_BYTES
        for _ in range(3):
            with writing(connection) as transaction:
                run(transaction, "INSERT INTO blobs(data) VALUES (x'00')")
        assert log_size(path) <= WAL_LIMIT_BYTES


def test_a_statement_s_changes_are_counted(tmp_path: Path) -> None:
    with database(tmp_path / "a.sqlite") as connection, writing(connection) as transaction:
        run_each(transaction, "INSERT INTO items(name) VALUES (?)", [("a",), ("b",)])
        assert run(transaction, "DELETE FROM items") == 2


def test_an_insert_that_made_no_row_says_so_rather_than_naming_the_last_row(tmp_path: Path) -> None:
    with database(tmp_path / "a.sqlite") as connection, writing(connection) as transaction:
        run(transaction, "INSERT INTO items(name) VALUES ('a')")
        with pytest.raises(ValueError, match="not one row was inserted"):
            inserted(transaction, "INSERT OR IGNORE INTO items(name) VALUES ('a')")


def test_an_error_sqlite_did_not_raise_has_no_code() -> None:
    assert primary_code(sqlite3.OperationalError("made here")) == 0


def insert_an_orphan(connection: sqlite3.Connection) -> None:
    with writing(connection) as transaction:
        run(transaction, "INSERT INTO children(parent) VALUES (1)")


def test_a_commit_that_fails_leaves_no_transaction_open_and_nothing_written(tmp_path: Path) -> None:
    with database(tmp_path / "a.sqlite") as connection:
        with writing(connection) as transaction:
            run(transaction, "CREATE TABLE children(parent INTEGER REFERENCES items(id) DEFERRABLE INITIALLY DEFERRED)")
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            insert_an_orphan(connection)
        assert not connection.in_transaction
        with reading(connection) as transaction:
            assert rows(transaction, Item, "SELECT parent, 'child' FROM children") == []


def fail_reading_after_sqlite_ended_the_transaction(connection: sqlite3.Connection) -> None:
    with reading(connection) as transaction:
        transaction.connection.execute("ROLLBACK")
        raise KeyError


def test_an_error_while_reading_is_not_hidden_by_the_end_of_the_reading(tmp_path: Path) -> None:
    with database(tmp_path / "a.sqlite") as connection:
        with pytest.raises(KeyError):
            fail_reading_after_sqlite_ended_the_transaction(connection)
        assert not connection.in_transaction


def read_damaged(connection: sqlite3.Connection) -> None:
    with reading(connection) as transaction:
        try:
            rows(transaction, Item, "SELECT id, name FROM items")
        except sqlite3.DatabaseError as error:
            raise KeyError(str(error)) from error


def test_a_reading_that_failed_ends_without_a_second_error(tmp_path: Path) -> None:
    path = tmp_path / "a.sqlite"
    with database(path) as connection, writing(connection) as transaction:
        run_each(transaction, "INSERT INTO items(name) VALUES (?)", [(str(n),) for n in range(500)])
    damage_table(path, "items")
    with contextlib.closing(connect(path, create=False)) as connection:
        with pytest.raises(KeyError, match="malformed"):
            read_damaged(connection)
        assert not connection.in_transaction


def test_a_reading_whose_transaction_ended_meanwhile_ends_quietly(tmp_path: Path) -> None:
    with database(tmp_path / "a.sqlite") as connection:
        with reading(connection) as transaction:
            transaction.connection.execute("ROLLBACK")
        assert not connection.in_transaction


def test_the_first_row_is_read_without_the_rest(tmp_path: Path) -> None:
    with database(tmp_path / "a.sqlite") as connection:
        with writing(connection) as transaction:
            run_each(transaction, "INSERT INTO items(name) VALUES (?)", [("a",), ("b",)])
        with reading(connection) as transaction:
            assert first(transaction, Item, "SELECT id, name FROM items ORDER BY id") == Item(1, "a")
            assert first(transaction, Item, "SELECT id, name FROM items WHERE id > 5") is None


def test_a_database_opens_at_a_path_of_spaces_marks_and_letters_of_any_script(tmp_path: Path) -> None:
    path = tmp_path / "my notes #1 ü%20 & 日本" / "a b.sqlite"
    with database(path), contextlib.closing(connect(path, create=False)) as again, reading(again) as transaction:
        assert rows(transaction, Item, "SELECT id, name FROM items") == []
    assert path.is_file()


def test_an_sqlite_older_than_sift_needs_is_refused_with_what_to_do() -> None:
    require_sqlite(MINIMUM_SQLITE)
    require_sqlite((MINIMUM_SQLITE[0], MINIMUM_SQLITE[1] + 1, 0))
    with pytest.raises(StoreUnavailableError, match=r"SQLite 3\.37\.2") as refused:
        require_sqlite((3, 37, 2))
    assert (
        refused.value.remedy
        == "run sift on a Python with SQLite 3.38.0 or newer, as uv, python.org and Homebrew install"
    )
