import os
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path
from typing import cast, final

import pytest

from sift.api.errors import Problem, StoreUnavailableError
from sift.storage.store import FILE_SUFFIXES, WRITER_SUFFIX
from sift.storage.writer import LockFile, lock_file, sole_writer

HOLD = (
    "import os, sys; from pathlib import Path; from sift.storage.writer import sole_writer\n"
    "with sole_writer(Path(sys.argv[1]), 'sift embed'):\n"
    "    print('held', os.getpid(), flush=True)\n"
    "    sys.stdin.readline()\n"
    "    os._exit(3)\n"
)


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


def test_writers_of_other_stores_go_on_side_by_side(tmp_path: Path) -> None:
    with sole_writer(tmp_path / "a.sqlite", "sift index"), sole_writer(tmp_path / "b.sqlite", "sift index"):
        pass


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
            holder.stdin.write("die\n")
            holder.stdin.flush()
            assert holder.wait(timeout=60) == 3
        finally:
            holder.kill()
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
class Failing:
    def __init__(self, real: LockFile) -> None:
        self.real = real

    def execute(self, sql: str, parameters: tuple[int | str, ...] = (), /) -> sqlite3.Cursor:
        if sql == "BEGIN IMMEDIATE":
            raise sqlite3.OperationalError("disk I/O error")
        return self.real.execute(sql, parameters)

    def close(self) -> None:
        self.real.close()


def failing(path: Path) -> LockFile:
    return Failing(lock_file(path))


def test_a_writer_file_that_cannot_be_locked_for_another_reason_says_that_reason(tmp_path: Path) -> None:
    with (
        pytest.raises(sqlite3.OperationalError, match="disk I/O error"),
        sole_writer(tmp_path / "notes.sqlite", "sift index", failing),
    ):
        pass
