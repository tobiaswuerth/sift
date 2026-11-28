import contextlib
import sqlite3
from pathlib import Path
from typing import NamedTuple

import pytest

from sift.api.errors import Problem, StoreUnavailableError
from sift.storage.database import connect, reading, rows, run, writing
from sift.storage.migrations import Migration, Older, migrated, readable_version, version, writable_version

FIRST = Migration(("CREATE TABLE a(x INTEGER) STRICT",))
SECOND = Migration(("CREATE TABLE b(y INTEGER) STRICT", "INSERT INTO b(y) VALUES (1)"))
READ_BY_OLDER = Migration(("CREATE TABLE c(z INTEGER) STRICT",), Older.READS)
WRITTEN_BY_OLDER = Migration(("CREATE INDEX a_x ON a(x)",), Older.WRITES)
ALSO_WRITTEN_BY_OLDER = Migration(("CREATE INDEX a_again ON a(x)",), Older.WRITES)


class Table(NamedTuple):
    name: str


class Marked(NamedTuple):
    read_from: int
    written_from: int


def tables(path: Path) -> tuple[int, list[str]]:
    with contextlib.closing(connect(path, create=False)) as connection, reading(connection) as transaction:
        found = rows(transaction, Table, "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        return version(transaction), [table.name for table in found if table.name != "compatibility"]


def migrate(path: Path, migrations: tuple[Migration, ...]) -> int:
    with contextlib.closing(connect(path, create=True)) as connection, writing(connection) as transaction:
        return migrated(transaction, migrations, path)


def marked(path: Path) -> list[Marked]:
    with contextlib.closing(connect(path, create=False)) as connection, reading(connection) as transaction:
        return rows(transaction, Marked, "SELECT read_from, written_from FROM compatibility")


def opened_knowing(path: Path, known: int) -> tuple[int, int]:
    with contextlib.closing(connect(path, create=False)) as connection, reading(connection) as transaction:
        return readable_version(transaction, known, path), writable_version(transaction, known, path)


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
    broken = Migration(("CREATE TABLE c(z INTEGER) STRICT", "INSERT INTO nowhere VALUES (1)"))
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        migrate(tmp_path / "a.sqlite", (FIRST, broken))
    assert tables(tmp_path / "a.sqlite") == (1, ["a"])
    assert marked(tmp_path / "a.sqlite") == [Marked(1, 1)]


def test_version_of_a_database_with_none_is_zero(tmp_path: Path) -> None:
    with contextlib.closing(connect(tmp_path / "a.sqlite", create=True)) as connection:
        with writing(connection) as transaction:
            run(transaction, "CREATE TABLE a(x INTEGER) STRICT")
        with reading(connection) as transaction:
            assert version(transaction) == 0


@pytest.mark.parametrize(
    "migrations, mark",
    [
        ((FIRST,), Marked(1, 1)),
        ((FIRST, SECOND), Marked(2, 2)),
        ((FIRST, READ_BY_OLDER), Marked(1, 2)),
        ((FIRST, WRITTEN_BY_OLDER), Marked(1, 1)),
        ((FIRST, SECOND, WRITTEN_BY_OLDER, READ_BY_OLDER, ALSO_WRITTEN_BY_OLDER), Marked(2, 4)),
    ],
)
def test_a_database_says_from_which_layout_on_a_sift_may_read_it_and_from_which_write_it(
    tmp_path: Path, migrations: tuple[Migration, ...], mark: Marked
) -> None:
    migrate(tmp_path / "a.sqlite", migrations[:1])
    migrate(tmp_path / "a.sqlite", migrations)
    assert marked(tmp_path / "a.sqlite") == [mark]


def test_an_older_sift_opens_what_a_newer_one_changed_in_a_way_it_can_live_with(tmp_path: Path) -> None:
    path = tmp_path / "a.sqlite"
    migrate(path, (FIRST, WRITTEN_BY_OLDER, ALSO_WRITTEN_BY_OLDER))
    assert opened_knowing(path, 1) == (3, 3)
    assert opened_knowing(path, 3) == (3, 3)
    assert opened_knowing(path, 5) == (3, 3)


def test_an_older_sift_reads_what_a_newer_one_keeps_in_a_way_it_cannot_keep_and_may_not_change_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "a.sqlite"
    migrate(path, (FIRST, READ_BY_OLDER))
    with contextlib.closing(connect(path, create=False)) as connection, reading(connection) as transaction:
        assert readable_version(transaction, 1, path) == 2
        with pytest.raises(StoreUnavailableError) as refused:
            writable_version(transaction, 1, path)
    assert refused.value.message == (
        f"{path} was changed by a newer sift (layout 2; this sift knows up to 1): this one can read it, not change it"
    )
    assert (refused.value.remedy, refused.value.code) == ("upgrade sift", Problem.STORE_NEWER)
    assert opened_knowing(path, 2) == (2, 2)


def test_an_older_sift_is_refused_what_a_newer_one_laid_out_anew(tmp_path: Path) -> None:
    path = tmp_path / "a.sqlite"
    migrate(path, (FIRST, WRITTEN_BY_OLDER, SECOND))
    for known in (1, 2):
        with pytest.raises(StoreUnavailableError, match="was written by a newer sift") as refused:
            opened_knowing(path, known)
        assert (refused.value.remedy, refused.value.code) == ("upgrade sift", Problem.STORE_NEWER)
    assert opened_knowing(path, 3) == (3, 3)


def test_a_database_numbered_past_this_sift_that_says_nothing_of_older_ones_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "a.sqlite"
    with contextlib.closing(connect(path, create=True)) as connection, writing(connection) as transaction:
        run(transaction, "CREATE TABLE a(x INTEGER) STRICT")
        run(transaction, "PRAGMA user_version = 12")
    with pytest.raises(StoreUnavailableError, match=r"layout 12; this sift knows up to 1"):
        opened_knowing(path, 1)
    migrate(tmp_path / "b.sqlite", (FIRST,))
    with contextlib.closing(connect(tmp_path / "b.sqlite", create=False)) as connection:
        with writing(connection) as transaction:
            run(transaction, "DELETE FROM compatibility")
            run(transaction, "PRAGMA user_version = 2")
        with pytest.raises(StoreUnavailableError, match="was written by a newer sift"):
            opened_knowing(tmp_path / "b.sqlite", 1)
