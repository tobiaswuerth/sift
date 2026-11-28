import os
import sqlite3
import threading
from collections.abc import Callable
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import final

import pytest

from sift.api.errors import Problem, StoreUnavailableError
from sift.storage.store_files import WRITER_SUFFIX
from sift.storage.writer import (
    ASK_AT_ONCE,
    HOLDER,
    NAMING,
    WAIT_A_MOMENT,
    WRITING,
    Holder,
    LockFile,
    held_by,
    lock_file,
    sole_writer,
)
from tests.waits import joined, waited


@final
class NamingLock:
    def __init__(self, real: LockFile, naming: threading.Event, asked: threading.Event) -> None:
        self.real, self.naming, self.asked = real, naming, asked

    def execute(self, sql: str, parameters: tuple[int | str, ...] = (), /) -> sqlite3.Cursor:
        if sql == HOLDER:
            self.naming.set()
            waited(self.asked)
        return self.real.execute(sql, parameters)

    def close(self) -> None:
        self.real.close()


@final
class ImpatientLock:
    def __init__(self, real: LockFile) -> None:
        self.real = real
        real.execute(ASK_AT_ONCE)

    def execute(self, sql: str, parameters: tuple[int | str, ...] = (), /) -> sqlite3.Cursor:
        return self.real.execute(ASK_AT_ONCE if sql == WAIT_A_MOMENT else sql, parameters)

    def close(self) -> None:
        self.real.close()


def impatient(path: Path) -> LockFile:
    return ImpatientLock(lock_file(path))


def test_a_writer_still_writing_its_name_is_not_named_as_the_writer_before_it(tmp_path: Path) -> None:
    store = tmp_path / "notes.sqlite"
    lock = Path(f"{store}{WRITER_SUFFIX}")
    with sole_writer(store, "an index"):
        pass
    naming, asked = threading.Event(), threading.Event()

    def slow() -> None:
        with sole_writer(store, "an embedding", lambda path: NamingLock(lock_file(path), naming, asked)):
            pass

    thread = threading.Thread(target=slow)
    thread.start()
    try:
        waited(naming)
        with pytest.raises(StoreUnavailableError) as refused, sole_writer(store, "a move", impatient):
            pass
        assert (refused.value.code, refused.value.message) == (Problem.BUSY, "another sift writes it now")
        assert held_by(lock, impatient) is None
    finally:
        asked.set()
        joined(thread, 30)
    assert held_by(lock) is None


@final
class OvertakenLock:
    def __init__(self, real: LockFile, overtaken: Callable[[], None]) -> None:
        self.real, self.overtaken, self.named = real, overtaken, False

    def execute(self, sql: str, parameters: tuple[int | str, ...] = (), /) -> sqlite3.Cursor:
        found = self.real.execute(sql, parameters)
        if sql == "COMMIT" and not self.named:
            self.named = True
            self.overtaken()
        return found

    def close(self) -> None:
        self.real.close()


def test_a_writer_another_named_itself_over_before_it_held_the_store_names_itself_again(tmp_path: Path) -> None:
    store = tmp_path / "notes.sqlite"
    lock = Path(f"{store}{WRITER_SUFFIX}")

    def named_by_another() -> None:
        with closing(lock_file(lock)) as other:
            other.execute(NAMING)
            other.execute("DELETE FROM holder")
            other.execute("INSERT INTO holder(pid, purpose) VALUES (?, ?)", (4242, "a move"))
            other.execute("COMMIT")

    with sole_writer(store, "an index", lambda path: OvertakenLock(lock_file(path), named_by_another)):
        assert held_by(lock, impatient) == Holder(os.getpid(), "an index")
    assert held_by(lock) is None


def test_a_writer_refused_after_it_named_itself_names_nobody_rather_than_itself(tmp_path: Path) -> None:
    store = tmp_path / "notes.sqlite"
    lock = Path(f"{store}{WRITER_SUFFIX}")
    with sole_writer(store, "an index"):
        pass
    with closing(lock_file(lock)) as first:

        def held_by_the_first() -> None:
            first.execute(WRITING)

        overtaken = partial(OvertakenLock, overtaken=held_by_the_first)
        with (
            pytest.raises(StoreUnavailableError) as refused,
            sole_writer(store, "a move", lambda path: overtaken(impatient(path))),
        ):
            pass
        first.execute("ROLLBACK")
    assert (refused.value.code, refused.value.message) == (Problem.BUSY, "another sift writes it now")


def test_a_writer_named_over_every_time_it_names_itself_gives_up_as_refused(tmp_path: Path) -> None:
    store = tmp_path / "notes.sqlite"
    lock = Path(f"{store}{WRITER_SUFFIX}")

    @final
    class AlwaysOvertaken:
        def __init__(self, real: LockFile) -> None:
            self.real = real

        def execute(self, sql: str, parameters: tuple[int | str, ...] = (), /) -> sqlite3.Cursor:
            found = self.real.execute(sql, parameters)
            if sql == "COMMIT":
                with closing(lock_file(lock)) as other:
                    other.execute(NAMING)
                    other.execute("UPDATE holder SET pid = 4242, purpose = 'a move'")
                    other.execute("COMMIT")
            return found

        def close(self) -> None:
            self.real.close()

    with (
        pytest.raises(StoreUnavailableError) as refused,
        sole_writer(store, "an index", lambda path: AlwaysOvertaken(lock_file(path))),
    ):
        pass
    assert (refused.value.code, refused.value.message) == (
        Problem.BUSY,
        "another sift writes it now (a move, process 4242)",
    )
    assert held_by(lock) is None
