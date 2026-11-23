import contextlib
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, cast

import pytest

from sift.api.errors import StoreUnavailableError
from sift.api.ids import CollectionId, first_row
from sift.storage.database import connect, first, reading, run, writing
from sift.storage.migrations import Migration
from sift.storage.store import (
    DAMAGED_REMEDY,
    LAYOUT,
    MISSING_REMEDY,
    connected,
    create_store,
    delete_store,
    open_store,
    read_store,
    stored_bytes,
    unavailable_if_unusable,
)
from tests.damage import damage_table

COLLECTION = CollectionId(7)


class Value(NamedTuple):
    value: str | int


def opened(
    path: Path, collection: CollectionId = COLLECTION, layout: Sequence[Migration] = LAYOUT
) -> contextlib.closing[sqlite3.Connection]:
    return contextlib.closing(open_store(path, collection, layout))


def test_a_new_store_records_its_layout_and_collection(tmp_path: Path) -> None:
    create_store(tmp_path / "s.sqlite", COLLECTION)
    with opened(tmp_path / "s.sqlite") as connection, reading(connection) as transaction:
        assert first(transaction, Value, "PRAGMA user_version") == Value(len(LAYOUT))
        assert first(transaction, Value, "SELECT value FROM meta WHERE key = 'collection'") == Value("7")


def test_a_collection_s_documents_and_chunks_are_numbered_from_its_first_row(tmp_path: Path) -> None:
    create_store(tmp_path / "s.sqlite", COLLECTION)
    with opened(tmp_path / "s.sqlite") as connection, writing(connection) as transaction:
        run(transaction, "INSERT INTO sources(path, modified_ns, size) VALUES ('a.md', 1, 1)")
        run(
            transaction,
            "INSERT INTO documents(source, key, digest, title, fields) VALUES (1, '', 'd', 't', '{}')",
        )
        run(
            transaction,
            "INSERT INTO chunks(document, first_line, last_line, start, length) VALUES (?, 1, 1, 0, 1)",
            (first_row(COLLECTION) + 1,),
        )
        assert first(transaction, Value, "SELECT max(id) FROM documents") == Value(first_row(COLLECTION) + 1)
        assert first(transaction, Value, "SELECT max(id) FROM chunks") == Value(first_row(COLLECTION) + 1)


def test_a_missing_store_is_not_made_anew(tmp_path: Path) -> None:
    with pytest.raises(StoreUnavailableError, match="missing") as refused:
        open_store(tmp_path / "gone.sqlite", COLLECTION)
    assert refused.value.remedy == MISSING_REMEDY
    assert list(tmp_path.iterdir()) == []


def test_a_file_that_is_no_database_is_a_damaged_store(tmp_path: Path) -> None:
    (tmp_path / "s.sqlite").write_bytes(b"not a database, just text that is long enough to have a header" * 10)
    with pytest.raises(StoreUnavailableError, match="damaged") as refused:
        open_store(tmp_path / "s.sqlite", COLLECTION)
    assert refused.value.remedy == DAMAGED_REMEDY


def test_a_database_that_is_no_store_is_refused(tmp_path: Path) -> None:
    with contextlib.closing(connect(tmp_path / "s.sqlite", create=True)) as connection, writing(connection) as tx:
        run(tx, "CREATE TABLE other(x INTEGER)")
    with pytest.raises(StoreUnavailableError, match="no store of sift"):
        open_store(tmp_path / "s.sqlite", COLLECTION)


def test_a_store_of_a_newer_sift_asks_for_an_upgrade(tmp_path: Path) -> None:
    create_store(tmp_path / "s.sqlite", COLLECTION)
    with contextlib.closing(connect(tmp_path / "s.sqlite", create=False)) as connection, writing(connection) as tx:
        run(tx, f"PRAGMA user_version = {len(LAYOUT) + 1}")
    with pytest.raises(StoreUnavailableError, match="newer sift") as refused:
        open_store(tmp_path / "s.sqlite", COLLECTION)
    assert refused.value.remedy == "upgrade sift"


def test_a_store_of_another_collection_is_refused(tmp_path: Path) -> None:
    create_store(tmp_path / "s.sqlite", CollectionId(3))
    with pytest.raises(StoreUnavailableError, match="store of collection 3, not of collection 7"):
        open_store(tmp_path / "s.sqlite", COLLECTION)


def test_a_store_that_names_no_collection_is_refused(tmp_path: Path) -> None:
    create_store(tmp_path / "s.sqlite", COLLECTION)
    with contextlib.closing(connect(tmp_path / "s.sqlite", create=False)) as connection, writing(connection) as tx:
        run(tx, "DELETE FROM meta")
    with pytest.raises(StoreUnavailableError, match="store of no collection"):
        open_store(tmp_path / "s.sqlite", COLLECTION)


def test_a_store_damaged_where_it_is_read_is_refused(tmp_path: Path) -> None:
    create_store(tmp_path / "s.sqlite", COLLECTION)
    raw = bytearray((tmp_path / "s.sqlite").read_bytes())
    raw[4096:] = b"\xff" * (len(raw) - 4096)
    (tmp_path / "s.sqlite").write_bytes(bytes(raw))
    with pytest.raises(StoreUnavailableError, match="damaged"):
        open_store(tmp_path / "s.sqlite", COLLECTION)


def test_an_error_that_is_no_damage_is_raised_as_it_is(tmp_path: Path) -> None:
    create_store(tmp_path / "s.sqlite", COLLECTION)
    with contextlib.closing(connect(tmp_path / "s.sqlite", create=False)) as connection, writing(connection) as tx:
        run(tx, "DROP TABLE meta")
    with pytest.raises(sqlite3.OperationalError, match="no such table: meta"):
        open_store(tmp_path / "s.sqlite", COLLECTION)


def test_a_store_is_never_made_over_a_file_that_is_there(tmp_path: Path) -> None:
    (tmp_path / "s.sqlite").write_bytes(b"someone's")
    with pytest.raises(FileExistsError):
        create_store(tmp_path / "s.sqlite", COLLECTION)
    assert (tmp_path / "s.sqlite").read_bytes() == b"someone's"


def test_a_store_that_could_not_be_made_leaves_no_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collection id"):
        create_store(tmp_path / "s.sqlite", CollectionId(0))
    assert list(tmp_path.iterdir()) == []


def test_deleting_a_store_deletes_its_database_log_and_shared_memory(tmp_path: Path) -> None:
    for name in ("s.sqlite", "s.sqlite-wal", "s.sqlite-shm", "other.sqlite"):
        (tmp_path / name).write_bytes(b"x")
    delete_store(tmp_path / "s.sqlite")
    assert [path.name for path in tmp_path.iterdir()] == ["other.sqlite"]


def test_deleting_a_store_that_is_not_there_does_nothing(tmp_path: Path) -> None:
    (tmp_path / "file").write_bytes(b"x")
    delete_store(tmp_path / "gone.sqlite")
    delete_store(tmp_path / "file" / "s.sqlite")
    assert [path.name for path in tmp_path.iterdir()] == ["file"]


def test_the_bytes_a_store_holds_count_its_log_and_not_its_shared_memory(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    assert stored_bytes(path) == path.stat().st_size
    with opened(path) as connection:
        with writing(connection) as transaction:
            run(transaction, "INSERT INTO sources(path, modified_ns, size) VALUES ('a.md', 1, 1)")
        log, shared = Path(f"{path}-wal"), Path(f"{path}-shm")
        assert log.stat().st_size > 0
        assert shared.is_file()
        assert stored_bytes(path) == path.stat().st_size + log.stat().st_size
    assert stored_bytes(tmp_path / "gone.sqlite") == 0


def test_damage_found_after_a_store_opened_is_said_as_that_of_a_store(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    damage_table(path, "sources")
    with (
        pytest.raises(StoreUnavailableError, match="damaged") as refused,
        unavailable_if_unusable(path),
        opened(path) as connection,
        reading(connection) as transaction,
    ):
        first(transaction, Value, "SELECT max(size) FROM sources")
    assert refused.value.remedy == DAMAGED_REMEDY


def test_an_error_that_is_no_damage_stays_the_error_it_is(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    with (
        pytest.raises(sqlite3.OperationalError, match="no such table"),
        unavailable_if_unusable(path),
        opened(path) as connection,
        reading(connection) as transaction,
    ):
        first(transaction, Value, "SELECT value FROM absent")


def test_a_store_another_writer_holds_is_unavailable_with_a_way_out(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    with opened(path) as holder, writing(holder):
        with (
            pytest.raises(StoreUnavailableError, match="another sift is writing to the store") as refused,
            unavailable_if_unusable(path),
            contextlib.closing(connect(path, create=False, timeout=0.05)) as waiting,
            writing(waiting),
        ):
            pass
        assert refused.value.remedy == "wait for it to finish, then run sift again"


def test_a_store_on_a_full_disk_is_unavailable_with_a_way_out(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    with opened(path) as connection:
        (pages,) = cast("tuple[int]", connection.execute("PRAGMA page_count").fetchone())
        connection.execute(f"PRAGMA max_page_count = {pages}")
        with (
            pytest.raises(StoreUnavailableError, match=r"the disk holding the store .* is full") as refused,
            unavailable_if_unusable(path),
            writing(connection) as transaction,
        ):
            run(transaction, "INSERT INTO meta(key, value) VALUES ('filler', ?)", ("x" * 100_000,))
    assert refused.value.remedy == "free space on it, then run sift again"


def test_a_store_of_an_older_layout_is_brought_up_to_this_one_when_opened(tmp_path: Path) -> None:
    create_store(tmp_path / "s.sqlite", COLLECTION)
    later = (*LAYOUT, ("CREATE TABLE added(x INTEGER) STRICT", "INSERT INTO added(x) VALUES (7)"))
    for _ in range(2):
        with opened(tmp_path / "s.sqlite", layout=later) as connection, reading(connection) as transaction:
            assert first(transaction, Value, "PRAGMA user_version") == Value(len(LAYOUT) + 1)
            assert first(transaction, Value, "SELECT sum(x) FROM added") == Value(7)
    with pytest.raises(StoreUnavailableError, match="newer sift"):
        open_store(tmp_path / "s.sqlite", COLLECTION)


def test_a_store_of_another_collection_is_not_brought_up_to_this_layout(tmp_path: Path) -> None:
    create_store(tmp_path / "s.sqlite", CollectionId(3))
    later = (*LAYOUT, ("CREATE TABLE added(x INTEGER) STRICT",))
    with pytest.raises(StoreUnavailableError, match="store of collection 3"):
        open_store(tmp_path / "s.sqlite", COLLECTION, later)
    with opened(tmp_path / "s.sqlite", CollectionId(3)) as connection, reading(connection) as transaction:
        assert first(transaction, Value, "PRAGMA user_version") == Value(len(LAYOUT))


def test_a_store_connected_or_read_is_closed_after_and_says_why_it_cannot_be_used(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    with connected(path, COLLECTION) as connection, writing(connection) as transaction:
        run(transaction, "INSERT INTO sources(path, modified_ns, size) VALUES ('a.md', 1, 1)")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with read_store(path, COLLECTION) as reading_it:
        assert first(reading_it, Value, "SELECT count(*) FROM sources") == Value(1)
    damage_table(path, "sources")
    with pytest.raises(StoreUnavailableError, match="damaged"), read_store(path, COLLECTION) as reading_it:
        first(reading_it, Value, "SELECT max(size) FROM sources")
    with pytest.raises(StoreUnavailableError, match="missing"), connected(tmp_path / "gone.sqlite", COLLECTION):
        pass
