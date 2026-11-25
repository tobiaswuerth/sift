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
from sift.storage.writer import (
    ASK_AGAIN,
    ASK_AT_ONCE,
    HELD,
    WAIT_A_MOMENT,
    Holder,
    LockFile,
    held_by,
    lock_file,
    sole_writer,
)
from tests.environments import clean_environment
from tests.waits import crossed, joined, waited

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
    with pytest.raises(StoreUnavailableError, match=r"the lock file .* could not be read or written: disk I/O error"):
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
    held, asking = threading.Event(), threading.Event()

    def moment() -> None:
        with closing(lock_file(Path(f"{store}{WRITER_SUFFIX}"))) as other:
            for statement in taken:
                other.execute(statement)
            held.set()
            waited(asking)
            other.execute("ROLLBACK")

    def opened(path: Path) -> LockFile:
        asking.set()
        return lock_file(path)

    thread = threading.Thread(target=moment)
    thread.start()
    waited(held)
    try:
        with sole_writer(store, "sift embed", opened):
            assert held_by(Path(f"{store}{WRITER_SUFFIX}")) == Holder(os.getpid(), "sift embed")
    finally:
        joined(thread, 30)


def test_a_writer_that_dies_holding_its_store_leaves_it_free(tmp_path: Path) -> None:
    store = tmp_path / "notes.sqlite"
    environment = clean_environment(PYTHONPATH=os.pathsep.join(sys.path))
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
                StoreUnavailableError, match=r"the lock file .* cannot be written: attempt to write a readonly database"
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


def disk_failed() -> sqlite3.OperationalError:
    error = sqlite3.OperationalError("disk I/O error")
    error.sqlite_errorcode, error.sqlite_errorname = sqlite3.SQLITE_IOERR_WRITE, "SQLITE_IOERR_WRITE"
    return error


@final
class RecordedLock:
    def __init__(self, real: LockFile, failing: str | None = None) -> None:
        self.real, self.failing, self.said = real, failing, list[str]()

    def execute(self, sql: str, parameters: tuple[int | str, ...] = (), /) -> sqlite3.Cursor:
        self.said.append(sql)
        if sql == self.failing:
            raise disk_failed()
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
    asked = [ASK_AT_ONCE, "BEGIN IMMEDIATE", ASK_AGAIN, "BEGIN IMMEDIATE"]
    assert opened[0].said == [*asked, WAIT_A_MOMENT, HELD, "SELECT pid, purpose FROM holder"]


@final
class ContendedLock:
    def __init__(self, real: LockFile, other: LockFile) -> None:
        self.real, self.other, self.begun = real, other, 0

    def execute(self, sql: str, parameters: tuple[int | str, ...] = (), /) -> sqlite3.Cursor:
        if sql == "BEGIN IMMEDIATE":
            self.begun += 1
            self.other.execute("BEGIN IMMEDIATE" if self.begun == 1 else "ROLLBACK")
        return self.real.execute(sql, parameters)

    def close(self) -> None:
        self.real.close()


def test_a_lock_another_asker_takes_for_a_moment_is_not_said_to_be_held(tmp_path: Path) -> None:
    store = tmp_path / "server"
    lock = Path(f"{store}{WRITER_SUFFIX}")
    with sole_writer(store, "sift serve"):
        pass
    with closing(lock_file(lock)) as other:
        contended = ContendedLock(lock_file(lock), other)
        assert (held_by(lock, lambda _path: contended), contended.begun) == (None, 2)


def test_a_writer_file_that_cannot_be_locked_for_another_reason_says_that_reason(tmp_path: Path) -> None:
    with (
        pytest.raises(
            StoreUnavailableError, match=r"the lock file .* could not be read or written: disk I/O"
        ) as refused,
        sole_writer(tmp_path / "notes.sqlite", "sift index", failing),
    ):
        pass
    assert refused.value.remedy == "check the drive it lies on"


def test_two_first_writers_of_a_store_take_it_in_turn_or_are_refused_naming_the_other(tmp_path: Path) -> None:
    outcomes = list[str]()

    def writer(store: Path, barrier: threading.Barrier, name: str) -> None:
        crossed(barrier, 30)
        try:
            with sole_writer(store, name):
                outcomes.append("held")
        except StoreUnavailableError as refused:
            outcomes.append(f"{refused.code}: {refused.message}"[:27])

    for trial in range(50):
        store, barrier = tmp_path / f"s{trial}.sqlite", threading.Barrier(2)
        threads = [threading.Thread(target=writer, args=(store, barrier, f"w{n}")) for n in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            joined(thread, 30)
    assert set(outcomes) <= {"held", "busy: another sift writes it"}
    assert (len(outcomes), outcomes.count("held") >= 50) == (100, True)


def test_who_holds_a_fresh_lock_is_asked_while_its_first_writer_makes_it(tmp_path: Path) -> None:
    lock = tmp_path / "s.sqlite.writer"
    lock.touch()
    with closing(lock_file(lock)) as first:
        first.execute("BEGIN IMMEDIATE")
        first.execute("CREATE TABLE IF NOT EXISTS holder(pid INTEGER NOT NULL, purpose TEXT NOT NULL) STRICT")
        assert held_by(lock) is None


TAKE_AT_ONCE = (
    "import sqlite3, sys\n"
    "connection = sqlite3.connect(sys.argv[1], timeout=0, isolation_level=None)\n"
    "try:\n"
    "    connection.execute('BEGIN IMMEDIATE')\n"
    "    print('taken')\n"
    "except sqlite3.OperationalError as error:\n"
    "    print(error)\n"
)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX locks belong to a process and are dropped by any close")
def test_asking_again_for_a_lock_this_process_holds_leaves_it_held_against_others(tmp_path: Path) -> None:
    store = tmp_path / "notes.sqlite"
    lock = Path(f"{store}{WRITER_SUFFIX}")
    with sole_writer(store, "sift index"):
        with pytest.raises(StoreUnavailableError, match="another sift writes it now"), sole_writer(store, "sift embed"):
            pass
        other = subprocess.run(
            [sys.executable, "-c", TAKE_AT_ONCE, str(lock)], capture_output=True, text=True, timeout=60, check=True
        )
    assert other.stdout == "database is locked\n"
