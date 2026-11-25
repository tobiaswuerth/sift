import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.ids import CollectionId
from sift.api.indexing import IndexCollections
from sift.api.links import FindSimilar
from sift.api.search import SearchHybrid, SearchVectors
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.app.query import query
from sift.app.resident import CodesAt, ResidentCodes
from sift.app.searching import Watching
from sift.app.similar import similar
from sift.app.vsearch import vsearch
from sift.retrieval.vector.nearest import Coded, coded
from tests.files import write
from tests.models import Hashed
from tests.waits import joined, still_running, waited


@dataclass(slots=True)
class Counting:
    held: ResidentCodes = field(default_factory=ResidentCodes)
    reads: list[CodesAt] = field(default_factory=list[CodesAt])
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, at: CodesAt, read: Callable[[], Coded]) -> Coded:
        def counted() -> Coded:
            with self.lock:
                self.reads.append(at)
            return read()

        return self.held(at, counted)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    write(tmp_path / "notes" / "halo.md", "# Halo\n\nDark matter halo models.\n")
    write(tmp_path / "notes" / "brush.md", "# Brush\n\nA robot plans brush strokes.\n")
    write(tmp_path / "papers" / "shaping.md", "# Shaping\n\nReward shaping speeds up learning.\n")
    home = tmp_path / "home"
    for name in ("notes", "papers"):
        add(home, AddCollection(str(tmp_path / name)))
    index(home, IndexCollections(), lambda _: None)
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    return home


def found(home: Path, counting: Counting | None = None) -> list[list[str]]:
    watching = Watching() if counting is None else Watching(codes=counting)
    return [
        [hit.id for hit in vsearch(home, SearchVectors("dark matter halo"), Hashed(), watching=watching).hits],
        [hit.id for hit in query(home, SearchHybrid("robot brush strokes"), Hashed(), watching=watching).hits],
        [hit.id for hit in similar(home, FindSimilar("notes/halo.md"), watching=watching).hits],
    ]


def test_searches_in_one_process_read_a_store_s_codes_once_and_find_what_reading_them_each_time_finds(
    home: Path,
) -> None:
    counting = Counting()
    assert found(home, counting) == found(home) == found(home, counting)
    assert sorted({str(at.store.name) for at in counting.reads}) == ["notes.sqlite", "papers.sqlite"]
    assert len(counting.reads) == 2


def test_codes_held_for_every_search_cannot_be_written(home: Path) -> None:
    counting = Counting()
    found(home, counting)
    for _, held in counting.held.held.values():
        for array in (held.ids, held.bits, held.words):
            with pytest.raises(ValueError, match="WRITEABLE"):
                array.flags.writeable = True


def test_codes_are_read_again_once_the_store_s_vectors_change(home: Path, tmp_path: Path) -> None:
    counting = Counting()
    found(home, counting)
    write(tmp_path / "notes" / "matter.md", "# Matter\n\nDark matter, and matter again.\n")
    index(home, IndexCollections(), lambda _: None)
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    after = found(home, counting)
    assert "notes/matter.md" in after[0]
    assert [at.store.name for at in counting.reads] == ["notes.sqlite", "papers.sqlite", "notes.sqlite"]
    assert counting.reads[0].generation < counting.reads[2].generation


def test_searches_at_once_read_each_store_s_codes_once(home: Path) -> None:
    counting, results, failures = Counting(), list[list[list[str]]](), list[BaseException]()

    def search() -> None:
        try:
            results.append(found(home, counting))
        except BaseException as error:
            failures.append(error)
            raise

    threads = [threading.Thread(target=search) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        joined(thread, 60)
    assert (failures, len(results), len(counting.reads)) == ([], 8, 2)
    assert all(each == results[0] for each in results)


def test_codes_held_for_one_store_are_handed_out_while_another_store_s_are_read() -> None:
    held, reading, read = ResidentCodes(), threading.Event(), threading.Event()
    one = CodesAt(Path("one.sqlite"), CollectionId(1), 1)
    other = CodesAt(Path("other.sqlite"), CollectionId(2), 1)
    codes = coded(np.arange(2, dtype=np.int64), np.zeros((2, 8), np.uint8), 64)
    assert held(one, lambda: codes) is codes

    def slow() -> Coded:
        reading.set()
        waited(read, 30)
        return codes

    thread = threading.Thread(target=held, args=(other, slow))
    thread.start()
    assert reading.wait(30)
    handed: list[Coded] = []
    beside = threading.Thread(target=lambda: handed.append(held(one, lambda: codes)))
    beside.start()
    joined(beside, 5)
    read.set()
    joined(thread, 30)
    assert handed == [codes]


def test_a_search_waiting_on_a_store_s_codes_takes_those_just_read() -> None:
    held, reading, read = ResidentCodes(), threading.Event(), threading.Event()
    at = CodesAt(Path("one.sqlite"), CollectionId(1), 1)
    codes = coded(np.arange(2, dtype=np.int64), np.zeros((2, 8), np.uint8), 64)
    reads: list[int] = []

    def slow() -> Coded:
        reads.append(1)
        reading.set()
        waited(read, 30)
        return codes

    first = threading.Thread(target=held, args=(at, slow))
    first.start()
    assert reading.wait(30)
    handed: list[Coded] = []
    second = threading.Thread(target=lambda: handed.append(held(at, slow)))
    second.start()
    still_running(second, 0.2)
    read.set()
    joined(first, 30)
    joined(second, 30)
    assert (handed, reads) == ([codes], [1])
