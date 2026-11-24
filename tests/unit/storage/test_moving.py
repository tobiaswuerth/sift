import errno
import sqlite3
import stat
from contextlib import closing
from pathlib import Path
from typing import NamedTuple

import pytest

from sift.api.errors import Problem, StoreUnavailableError
from sift.api.ids import CollectionId
from sift.storage.database import reading, rows, run, writing
from sift.storage.moving import Transfer, held_alone, transferred
from sift.storage.store import create_store, open_store

COLLECTION = CollectionId(3)


class Source(NamedTuple):
    path: str


def store_of(folder: Path) -> Path:
    path = folder / "s.sqlite"
    create_store(path, COLLECTION)
    with closing(open_store(path, COLLECTION)) as connection, writing(connection) as transaction:
        for n in range(300):
            run(
                transaction,
                "INSERT INTO sources(path, modified_ns, size) VALUES (?, 1, 1)",
                (f"{n:04}-{'x' * 200}.md",),
            )
    return path


def held(path: Path) -> list[Source]:
    with closing(open_store(path, COLLECTION)) as connection, reading(connection) as transaction:
        return rows(transaction, Source, "SELECT path FROM sources ORDER BY id")


def refusing(code: int) -> Transfer:
    def link(_old: Path, _new: Path) -> None:
        raise OSError(code, "no link")

    return Transfer(link=link, pages=2)


def test_a_store_held_alone_keeps_every_other_sift_out_and_holds_its_log(tmp_path: Path) -> None:
    path = store_of(tmp_path)
    with held_alone(path) as connection:
        assert connection.execute("PRAGMA wal_checkpoint").fetchone()[1] in (0, -1)
        with closing(sqlite3.connect(path, timeout=0)) as other, pytest.raises(sqlite3.OperationalError):
            other.execute("SELECT count(*) FROM sources").fetchone()
    assert len(held(path)) == 300


def test_a_store_another_sift_reads_now_is_not_held(tmp_path: Path) -> None:
    path = store_of(tmp_path)
    with closing(sqlite3.connect(path, isolation_level=None)) as reader:
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM sources").fetchone()
        with pytest.raises(StoreUnavailableError) as refused, held_alone(path):
            pass
    assert refused.value.code is Problem.BUSY
    assert "another sift has the store" in refused.value.message


def test_a_store_that_may_not_be_written_is_not_held_and_says_so(tmp_path: Path) -> None:
    path = store_of(tmp_path)
    path.chmod(stat.S_IREAD)
    try:
        with pytest.raises(StoreUnavailableError, match="could not be read or written") as refused, held_alone(path):
            pass
        assert refused.value.code is Problem.STORE_UNAVAILABLE
    finally:
        path.chmod(stat.S_IREAD | stat.S_IWRITE)
    assert len(held(path)) == 300


def test_a_store_moves_by_a_second_name_on_its_own_drive(tmp_path: Path) -> None:
    path = store_of(tmp_path)
    new = tmp_path / "elsewhere" / "t.sqlite"
    with held_alone(path) as connection:
        assert transferred(connection, path, new, lambda _done, _total: None) is False
    assert held(new) == held(path)


def test_a_store_on_another_drive_is_copied_page_by_page_with_its_progress(tmp_path: Path) -> None:
    path = store_of(tmp_path)
    new = tmp_path / "other drive" / "s.sqlite"
    seen: list[tuple[int, int]] = []
    with held_alone(path) as connection:
        assert transferred(connection, path, new, lambda done, total: seen.append((done, total)), refusing(errno.EXDEV))
    assert held(new) == held(path)
    assert len(seen) > 1
    assert seen[-1][0] == seen[-1][1] == new.stat().st_size
    assert [file.name for file in new.parent.iterdir()] == ["s.sqlite"]


def test_a_copy_stopped_leaves_nothing_at_its_new_place(tmp_path: Path) -> None:
    path = store_of(tmp_path)
    new = tmp_path / "other drive" / "s.sqlite"

    def stopped(done: int, _total: int) -> None:
        if done:
            raise KeyboardInterrupt

    with held_alone(path) as connection, pytest.raises(KeyboardInterrupt):
        transferred(connection, path, new, stopped, refusing(errno.EXDEV))
    assert list(new.parent.iterdir()) == []
    assert len(held(path)) == 300


def test_a_link_refused_for_another_reason_says_that_reason(tmp_path: Path) -> None:
    path = store_of(tmp_path)
    with held_alone(path) as connection, pytest.raises(OSError, match="no link"):
        transferred(connection, path, tmp_path / "t.sqlite", lambda _done, _total: None, refusing(errno.ENOSPC))
    assert not (tmp_path / "t.sqlite").exists()
