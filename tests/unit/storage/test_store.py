import contextlib
import sqlite3
import stat
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, cast

import pytest

from sift.api.errors import Problem, StoreUnavailableError
from sift.api.ids import CollectionId, first_row
from sift.storage.database import CACHE_KIBIBYTES, READ_CACHE_KIBIBYTES, Writing, connect, first, reading, run, writing
from sift.storage.layout import LAYOUT
from sift.storage.migrations import Migration, Older
from sift.storage.store import MISSING_REMEDY, connected, create_store, open_store, read_store
from sift.storage.store_files import stored_bytes
from sift.storage.unusable import DAMAGED_REMEDY, unavailable_if_unusable
from sift.storage.writer import sole_writer
from tests.damage import damage_table
from tests.waits import joined, still_running

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
        run(transaction, "INSERT INTO sources(path, modified_ns, size, folded) VALUES ('a.md', 1, 1, 'a.md')")
        run(
            transaction,
            "INSERT INTO documents(source, key, digest, title, fields) VALUES (1, '', 'd', 't', '{}')",
        )
        run(
            transaction,
            """INSERT INTO chunks(document, first_line, last_line, start, length, place)
               VALUES (?, 1, 1, 0, 1, (SELECT places + 1 FROM counts))""",
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
    assert refused.value.remedy == "remove the collection, then add its folder and index it again"


def test_a_database_that_is_no_store_is_refused(tmp_path: Path) -> None:
    with contextlib.closing(connect(tmp_path / "s.sqlite", create=True)) as connection, writing(connection) as tx:
        run(tx, "CREATE TABLE other(x INTEGER)")
    with pytest.raises(StoreUnavailableError, match="no store of sift"):
        open_store(tmp_path / "s.sqlite", COLLECTION)


def test_a_store_a_newer_sift_laid_out_anew_asks_for_an_upgrade(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION, (*LAYOUT, Migration(("CREATE TABLE added(x INTEGER) STRICT",))))
    with pytest.raises(StoreUnavailableError, match="was written by a newer sift") as refused:
        open_store(path, COLLECTION)
    assert (refused.value.remedy, refused.value.code) == ("upgrade sift", Problem.STORE_NEWER)
    with pytest.raises(StoreUnavailableError, match="was written by a newer sift"), read_store(path, COLLECTION):
        pass


def test_a_store_a_newer_sift_changed_in_a_way_this_one_can_live_with_is_read_and_written_as_before(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.sqlite"
    later = (*LAYOUT, Migration(("CREATE INDEX added ON sources(size)",), Older.WRITES))
    create_store(path, COLLECTION, later)
    with opened(path) as connection, writing(connection) as transaction:
        run(transaction, "INSERT INTO sources(path, modified_ns, size, folded) VALUES ('a.md', 1, 1, 'a.md')")
    with read_store(path, COLLECTION) as transaction:
        assert first(transaction, Value, "SELECT count(*) FROM sources") == Value(1)
        assert first(transaction, Value, "PRAGMA user_version") == Value(len(later))


def test_a_store_a_newer_sift_keeps_in_a_way_this_one_cannot_keep_is_read_and_not_changed(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION, (*LAYOUT, Migration(("CREATE TABLE added(x INTEGER) STRICT",), Older.READS)))
    with read_store(path, COLLECTION) as transaction:
        assert first(transaction, Value, "SELECT value FROM meta WHERE key = 'collection'") == Value(str(COLLECTION))
    with pytest.raises(StoreUnavailableError, match="this one can read it, not change it") as refused:
        open_store(path, COLLECTION)
    assert (refused.value.remedy, refused.value.code) == ("upgrade sift", Problem.STORE_NEWER)
    with pytest.raises(StoreUnavailableError, match="this one can read it, not change it"), connected(path, COLLECTION):
        pass


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


def test_a_store_is_never_made_over_a_file_that_is_there(tmp_path: Path) -> None:
    (tmp_path / "s.sqlite").write_bytes(b"someone's")
    with pytest.raises(FileExistsError):
        create_store(tmp_path / "s.sqlite", COLLECTION)
    assert (tmp_path / "s.sqlite").read_bytes() == b"someone's"


def test_the_log_of_a_store_that_is_gone_is_not_read_into_the_store_made_where_it_was(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    with connected(path, COLLECTION) as connection:
        with writing(connection) as transaction:
            run(transaction, "INSERT INTO meta(key, value) VALUES ('left', 'by the store before')")
        left = Path(f"{path}-wal").read_bytes()
    for file in tmp_path.iterdir():
        file.unlink()
    Path(f"{path}-wal").write_bytes(left)
    create_store(path, CollectionId(8))
    with read_store(path, CollectionId(8)) as transaction:
        assert first(transaction, Value, "SELECT value FROM meta WHERE key = 'left'") is None


def test_a_store_that_could_not_be_made_leaves_no_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collection id"):
        create_store(tmp_path / "s.sqlite", CollectionId(0))
    assert list(tmp_path.iterdir()) == []


def test_the_bytes_a_store_holds_count_its_log_and_not_its_shared_memory(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    assert stored_bytes(path) == path.stat().st_size
    with opened(path) as connection:
        with writing(connection) as transaction:
            run(transaction, "INSERT INTO sources(path, modified_ns, size, folded) VALUES ('a.md', 1, 1, 'a.md')")
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
    later = (*LAYOUT, Migration(("CREATE TABLE added(x INTEGER) STRICT", "INSERT INTO added(x) VALUES (7)")))
    for _ in range(2):
        with opened(tmp_path / "s.sqlite", layout=later) as connection, reading(connection) as transaction:
            assert first(transaction, Value, "PRAGMA user_version") == Value(len(LAYOUT) + 1)
            assert first(transaction, Value, "SELECT sum(x) FROM added") == Value(7)
    with pytest.raises(StoreUnavailableError, match="newer sift"):
        open_store(tmp_path / "s.sqlite", COLLECTION)


def test_a_store_two_open_at_once_is_brought_up_to_this_layout_once(tmp_path: Path) -> None:
    store, ran, found = tmp_path / "s.sqlite", list[str](), list[Value | None]()
    create_store(store, COLLECTION)

    def backfilled(_: Writing) -> None:
        ran.append("backfill")

    later = (*LAYOUT, Migration(("CREATE TABLE added(x INTEGER) STRICT", backfilled)))

    def open_it() -> None:
        with opened(store, layout=later) as connection, reading(connection) as transaction:
            found.append(first(transaction, Value, "PRAGMA user_version"))

    both = [threading.Thread(target=open_it) for _ in range(2)]
    with contextlib.closing(connect(store, create=False)) as holder, writing(holder):
        for each in both:
            each.start()
        for each in both:
            still_running(each, 0.3)
    for each in both:
        joined(each)
    assert (found, ran) == ([Value(len(later))] * 2, ["backfill"])


def test_a_store_of_another_collection_is_not_brought_up_to_this_layout(tmp_path: Path) -> None:
    create_store(tmp_path / "s.sqlite", CollectionId(3))
    later = (*LAYOUT, Migration(("CREATE TABLE added(x INTEGER) STRICT",)))
    with pytest.raises(StoreUnavailableError, match="store of collection 3"):
        open_store(tmp_path / "s.sqlite", COLLECTION, later)
    with opened(tmp_path / "s.sqlite", CollectionId(3)) as connection, reading(connection) as transaction:
        assert first(transaction, Value, "PRAGMA user_version") == Value(len(LAYOUT))


def test_a_store_connected_or_read_is_closed_after_and_says_why_it_cannot_be_used(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    with connected(path, COLLECTION) as connection, writing(connection) as transaction:
        run(transaction, "INSERT INTO sources(path, modified_ns, size, folded) VALUES ('a.md', 1, 1, 'a.md')")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with read_store(path, COLLECTION) as reading_it:
        assert first(reading_it, Value, "SELECT count(*) FROM sources") == Value(1)
    damage_table(path, "sources")
    with pytest.raises(StoreUnavailableError, match="damaged"), read_store(path, COLLECTION) as reading_it:
        first(reading_it, Value, "SELECT max(size) FROM sources")
    with pytest.raises(StoreUnavailableError, match="missing"), connected(tmp_path / "gone.sqlite", COLLECTION):
        pass


def test_a_store_of_this_layout_is_read_without_its_lock_and_one_of_an_older_layout_under_it(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    later = (*LAYOUT, Migration(("CREATE TABLE added(x INTEGER) STRICT", "INSERT INTO added(x) VALUES (7)")))
    with sole_writer(path, "sift index"):
        with read_store(path, COLLECTION) as transaction:
            assert first(transaction, Value, "PRAGMA user_version") == Value(len(LAYOUT))
        with (
            pytest.raises(StoreUnavailableError, match="another sift writes it now"),
            read_store(path, COLLECTION, later),
        ):
            pass
    with read_store(path, COLLECTION, later) as transaction:
        assert first(transaction, Value, "SELECT sum(x) FROM added") == Value(7)


def test_a_reader_keeps_a_page_cache_a_sixteenth_of_a_writer_s(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    with read_store(path, COLLECTION) as transaction, opened(path) as written:
        caches = [each.execute("PRAGMA cache_size").fetchone() for each in (transaction.connection, written)]
    assert caches == [(-READ_CACHE_KIBIBYTES,), (-CACHE_KIBIBYTES,)] == [(-(16 << 10),), (-(256 << 10),)]


def test_what_reads_a_store_cannot_change_it(tmp_path: Path) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    with pytest.raises(StoreUnavailableError, match="cannot be written"), read_store(path, COLLECTION) as transaction:
        transaction.connection.execute("DELETE FROM meta")
    with read_store(path, COLLECTION) as transaction:
        assert first(transaction, Value, "SELECT value FROM meta WHERE key = 'collection'") == Value(str(COLLECTION))


def test_a_store_that_may_only_be_read_is_read_and_left_as_it_was_unless_it_is_of_an_older_layout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    before = path.read_bytes()
    path.chmod(stat.S_IREAD)
    try:
        with read_store(path, COLLECTION) as transaction:
            assert first(transaction, Value, "PRAGMA user_version") == Value(len(LAYOUT))
        assert path.read_bytes() == before
        later = (*LAYOUT, Migration(("CREATE TABLE added(x INTEGER) STRICT",)))
        with pytest.raises(StoreUnavailableError, match="is marked read-only"), read_store(path, COLLECTION, later):
            pass
    finally:
        path.chmod(stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.parametrize(
    "sql, lacked",
    [
        ("DROP TABLE meta", "it lacks the table meta"),
        ("DROP TABLE links", "it lacks the table links"),
        ("DROP TRIGGER chunks_added", "it lacks the trigger chunks_added"),
        ("DROP VIEW keyword_documents", "it lacks the view keyword_documents"),
        ("DELETE FROM counts", "it lacks its counts"),
    ],
)
def test_a_store_lacking_what_its_layout_makes_is_damaged(tmp_path: Path, sql: str, lacked: str) -> None:
    path = tmp_path / "s.sqlite"
    create_store(path, COLLECTION)
    with contextlib.closing(connect(path, create=False)) as connection, writing(connection) as transaction:
        run(transaction, sql)
    with pytest.raises(StoreUnavailableError, match=lacked) as refused:
        open_store(path, COLLECTION)
    assert (refused.value.code, refused.value.remedy) == (Problem.STORE_DAMAGED, DAMAGED_REMEDY)
    with pytest.raises(StoreUnavailableError, match=lacked), read_store(path, COLLECTION):
        pass
