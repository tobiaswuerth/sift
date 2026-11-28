import json
import sqlite3
import threading
from collections.abc import Callable, Generator
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest

from sift.api.errors import StoreUnavailableError
from sift.storage.readers import STUCK_AHEAD_REMEDY, Ahead, ReadPool, readers_of
from tests.waits import crossed, waited

ROWS = "SELECT count(*) FROM rows WHERE id IN (SELECT value FROM json_each(?))"
MET = """WITH part(keys) AS (SELECT ?)
         SELECT met(keys) + (SELECT count(*) FROM rows WHERE id IN (SELECT value FROM json_each(keys))) FROM part"""

type Opener = Callable[[Path, int], ReadPool]


@pytest.fixture
def store(tmp_path: Path) -> Path:
    path = tmp_path / "store.sqlite"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE rows(id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
        connection.executemany("INSERT INTO rows VALUES (?, ?)", [(n, f"row {n}") for n in range(100)])
    return path


def with_met(met: Callable[[str], int]) -> Opener:
    def opened(path: Path, count: int) -> ReadPool:
        readers = readers_of(path, count)
        for connection in readers.every:
            connection.create_function("met", 1, met)
        return readers

    return opened


def test_rows_read_ahead_are_split_among_readers_that_read_them_at_the_same_time(store: Path) -> None:
    parts, together = list[list[int]](), threading.Barrier(3)

    def met(part: str) -> int:
        parts.append(cast("list[int]", json.loads(part)))
        crossed(together)
        return 0

    ahead = Ahead(store, parts=3, fewest=1, opened=with_met(met))
    try:
        ahead.touched(MET, list(range(8)))
        ahead.touched(MET, [40, 41, 42])
    finally:
        ahead.close()
    assert sorted(parts) == [[0, 1, 2], [3, 4, 5], [6, 7], [40], [41], [42]]


def test_a_few_rows_are_left_to_the_search_to_read(store: Path) -> None:
    opened = list[int]()

    def counted(path: Path, count: int) -> ReadPool:
        opened.append(count)
        return readers_of(path, count)

    ahead = Ahead(store, parts=4, fewest=16, opened=counted)
    ahead.touched(ROWS, list(range(15)))
    ahead.close()
    assert opened == []


def test_readers_are_opened_once_and_let_go_on_closing(store: Path) -> None:
    held = list[ReadPool]()

    def kept(path: Path, count: int) -> ReadPool:
        held.append(readers_of(path, count))
        return held[-1]

    ahead = Ahead(store, parts=2, fewest=1, opened=kept)
    for _ in range(3):
        ahead.touched(ROWS, list(range(10)))
    ahead.close()
    ahead.close()
    (readers,) = held
    assert len(readers.every) == 2
    for connection in readers.every:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_a_read_ahead_that_fails_says_so(store: Path) -> None:
    ahead = Ahead(store, parts=2, fewest=1)
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            ahead.touched("SELECT count(*) FROM gone WHERE id IN (SELECT value FROM json_each(?))", [1, 2, 3])
    finally:
        ahead.close()


@pytest.fixture
def stuck() -> Generator[threading.Event]:
    let_go = threading.Event()
    yield let_go
    let_go.set()


def test_rows_not_read_ahead_in_time_say_reading_the_store_is_stuck(store: Path, stuck: threading.Event) -> None:
    entered = threading.Event()

    def met(_part: str) -> int:
        entered.set()
        waited(stuck)
        return 0

    ahead = Ahead(store, parts=1, fewest=1, opened=with_met(met), waited=0.05)
    with pytest.raises(StoreUnavailableError, match="not read within 0 s") as refused:
        ahead.touched(MET, [1, 2])
    waited(entered)
    stuck.set()
    ahead.close()
    assert refused.value.remedy == STUCK_AHEAD_REMEDY
