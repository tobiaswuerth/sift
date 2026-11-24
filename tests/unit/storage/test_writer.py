import os
import sqlite3
import stat
import subprocess
import sys
import threading
from contextlib import closing
from pathlib import Path
from typing import cast, final

import pytest

from sift.api.errors import Problem, StoreUnavailableError
from sift.storage.store_files import FILE_SUFFIXES, WRITER_SUFFIX
from sift.storage.writer import ASK_AT_ONCE, WAIT_A_MOMENT, Holder, LockFile, held_by, lock_file, sole_writer

HOLD = (
    "import os, sys; from pathlib import Path; from sift.storage.writer import sole_writer\n"
    "with sole_writer(Path(sys.argv[1]), 'sift embed'):\n"
    "    print('held', os.getpid(), flush=True)\n"
    "    sys.stdin.readline()\n"
    "    os._exit(3)\n"
)
MOMENTS = (("BEGIN IMMEDIATE",), ("BEGIN", "SELECT pid FROM holder"))


def test_a_second_writer_of_a_store_is_refused_at_once_naming_the_first(tmp_path: Path) -> None:
    store = tmp_path / "notes.sqlite"
    with sole_writer(store, "sift index"):
        with (
            pytest.raises(StoreUnavailableError, match="another sift writes it now") as refused,
            sole_writer(store, "sift embed"),
        ):
            pass
        assert refused.value.code == Problem.BUSY
        assert f"sift index, process {os.getpid()}" in refused.value.message
        assert refused.value.remedy == "wait until it is done, or stop that process"
    with sole_writer(store, "sift embed"):
        pass


def test_who_holds_a_lock_is_told_without_taking_it_and_no_one_once_it_is_let_go(tmp_path: Path) -> None:
    store, lock = tmp_path / "home" / "server", tmp_path / "home" / "server.writer"
    assert held_by(lock) is None
    with sole_writer(store, "sift serve"):
        assert held_by(lock) == Holder(os.getpid(), "sift serve")
        assert held_by(lock) == Holder(os.getpid(), "sift serve")
    assert held_by(lock) is None
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        held_by(lock, failing)


def test_writers_of_other_stores_go_on_side_by_side(tmp_path: Path) -> None:
    with sole_writer(tmp_path / "a.sqlite", "sift index"), sole_writer(tmp_path / "b.sqlite", "sift index"):
        pass


@pytest.mark.parametrize("taken", MOMENTS)
def test_a_writer_takes_its_store_once_another_held_its_lock_only_for_a_moment(
    tmp_path: Path, taken: tuple[str, ...]
) -> None:
    store = tmp_path / "notes.sqlite"
    with sole_writer(store, "sift index"):
        pass
    held = threading.Event()

    def moment() -> None:
        with closing(lock_file(Path(f"{store}{WRITER_SUFFIX}"))) as other:
            for statement in taken:
                other.execute(statement)
            held.set()
            threading.Event().wait(0.1)
            other.execute("ROLLBACK")

    thread = threading.Thread(target=moment)
    thread.start()
    held.wait(30)
    try:
        with sole_writer(store, "sift embed"):
            assert held_by(Path(f"{store}{WRITER_SUFFIX}")) == Holder(os.getpid(), "sift embed")
    finally:
        thread.join(30)


def test_a_writer_that_dies_holding_its_store_leaves_it_free(tmp_path: Path) -> None:
    store = tmp_path / "notes.sqlite"
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    with subprocess.Popen(
        [sys.executable, "-c", HOLD, str(store)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
    ) as holder:
        try:
            assert holder.stdin is not None
            assert holder.stdout is not None
            said, pid = cast("str", holder.stdout.readline()).split()
            assert said == "held"
            with (
                pytest.raises(StoreUnavailableError, match=f"sift embed, process {pid}"),
                sole_writer(store, "sift index"),
            ):
                pass
            assert held_by(Path(f"{store}{WRITER_SUFFIX}")) == Holder(int(pid), "sift embed")
            holder.stdin.write("die\n")
            holder.stdin.flush()
            assert holder.wait(timeout=60) == 3
        finally:
            holder.kill()
    assert held_by(Path(f"{store}{WRITER_SUFFIX}")) is None
    with sole_writer(store, "sift index"):
        pass


def test_a_store_whose_writer_file_may_not_be_written_is_said_to_be_so(tmp_path: Path) -> None:
    store = tmp_path / "notes.sqlite"
    with sole_writer(store, "sift index"):
        pass
    held = Path(f"{store}.writer")
    held.chmod(stat.S_IREAD)
    try:
        with (
            pytest.raises(
                StoreUnavailableError, match="cannot be written: attempt to write a readonly database"
            ) as refused,
            sole_writer(store, "sift index"),
        ):
            pass
        assert refused.value.code == Problem.STORE_UNAVAILABLE
    finally:
        held.chmod(stat.S_IREAD | stat.S_IWRITE)


def test_a_writer_leaves_one_file_beside_the_store(tmp_path: Path) -> None:
    store = tmp_path / "notes.sqlite"
    with sole_writer(store, "sift index"):
        pass
    assert sorted(path.name for path in tmp_path.iterdir()) == ["notes.sqlite.writer"]
    assert WRITER_SUFFIX in FILE_SUFFIXES


@final
class RecordedLock:
    def __init__(self, real: LockFile, failing: str | None = None) -> None:
        self.real, self.failing, self.said = real, failing, list[str]()

    def execute(self, sql: str, parameters: tuple[int | str, ...] = (), /) -> sqlite3.Cursor:
        self.said.append(sql)
        if sql == self.failing:
            raise sqlite3.OperationalError("disk I/O error")
        return self.real.execute(sql, parameters)

    def close(self) -> None:
        self.real.close()


def failing(path: Path) -> LockFile:
    return RecordedLock(lock_file(path), "BEGIN IMMEDIATE")


def test_who_holds_a_lock_is_asked_at_once_and_read_waiting_only_a_moment(tmp_path: Path) -> None:
    store, opened = tmp_path / "server", list[RecordedLock]()

    def recorded(path: Path) -> LockFile:
        opened.append(RecordedLock(lock_file(path)))
        return opened[-1]

    with sole_writer(store, "sift serve"):
        assert held_by(Path(f"{store}{WRITER_SUFFIX}"), recorded) == Holder(os.getpid(), "sift serve")
    assert opened[0].said == [ASK_AT_ONCE, "BEGIN IMMEDIATE", WAIT_A_MOMENT, "SELECT pid, purpose FROM holder"]


def test_a_writer_file_that_cannot_be_locked_for_another_reason_says_that_reason(tmp_path: Path) -> None:
    with (
        pytest.raises(sqlite3.OperationalError, match="disk I/O error"),
        sole_writer(tmp_path / "notes.sqlite", "sift index", failing),
    ):
        pass
