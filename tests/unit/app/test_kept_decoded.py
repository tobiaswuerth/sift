import dataclasses
import threading
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from sift.api.ids import CollectionId
from sift.api.links import FindMentions
from sift.api.search import SearchHybrid, SearchKeywords
from sift.app.kept_decoded import ENTRY_BYTES, Decoded, KeptDecoded
from sift.app.mentions import mentions
from sift.app.query import query
from sift.app.resident import KeywordsAt
from sift.app.search import search
from sift.app.searching import Watching
from sift.retrieval.keyword.lists import Dense, PostingArrays
from tests.files import write
from tests.homes import home_of, indexed
from tests.models import Hashed
from tests.waits import joined

A = KeywordsAt(Path("a.sqlite"), CollectionId(2), 1)
B = KeywordsAt(Path("b.sqlite"), CollectionId(3), 1)


def arrays(n: int) -> PostingArrays:
    return np.arange(n, dtype=np.int64), np.ones(n, np.int64)


def test_decoded_blocks_are_kept_to_a_budget_of_bytes_the_longest_unused_let_go_first() -> None:
    kept = KeptDecoded(budget=3 * (320 + ENTRY_BYTES) + 10)
    blocks = kept(A).blocks
    for term in ("a", "b", "c"):
        blocks[1, term, 0] = arrays(20)
    assert blocks.get((1, "a", 0)) is not None
    blocks[1, "d", 0] = arrays(20)
    held = [term for term in "abcd" if blocks.get((1, term, 0)) is not None]
    assert (held, kept.size) == (["a", "c", "d"], 3 * (320 + ENTRY_BYTES))


def test_a_block_larger_than_the_budget_is_not_kept() -> None:
    kept = KeptDecoded(budget=100 + ENTRY_BYTES)
    blocks = kept(A).blocks
    blocks[1, "small", 0] = arrays(2)
    blocks[1, "large", 0] = arrays(20)
    assert (blocks.get((1, "large", 0)), kept.size) == (None, 32 + ENTRY_BYTES)


def test_what_a_store_held_at_another_generation_is_let_go_and_other_stores_keep_theirs() -> None:
    kept = KeptDecoded(budget=1 << 20)
    kept(A).blocks[1, "x", 0] = arrays(4)
    kept(B).blocks[1, "x", 0] = arrays(4)
    later = kept(dataclasses.replace(A, generation=2)).blocks
    held = (later.get((1, "x", 0)), kept(B).blocks.get((1, "x", 0)) is not None, kept.size)
    assert held == (None, True, 64 + ENTRY_BYTES)
    kept.kept_only({(B.store, B.collection)})
    kept.kept_only(set())
    assert (kept(B).blocks.get((1, "x", 0)), kept.size) == (None, 0)


def test_what_a_search_of_an_earlier_generation_decodes_is_never_read_at_a_later_one() -> None:
    kept = KeptDecoded(budget=1 << 20)
    earlier = kept(A).blocks
    later = kept(dataclasses.replace(A, generation=2)).blocks
    earlier[1, "x", 0] = arrays(4)
    assert (later.get((1, "x", 0)), kept.size) == (None, 0)


def test_searches_side_by_side_keep_the_budget_and_count_what_they_hold() -> None:
    kept = KeptDecoded(budget=4000)

    def searching(worker: int) -> None:
        for n in range(300):
            at = dataclasses.replace(A, generation=1 + n % 3) if worker % 2 else B
            blocks = kept(at).blocks
            blocks[worker, "t", n % 17] = arrays(1 + n % 9)
            blocks.get((worker, "t", (n * 7) % 17))

    threads = [threading.Thread(target=searching, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        joined(thread, 60)
    assert 0 < kept.size <= 4000
    assert kept.size == sum(size for _, size in kept.entries.values())


@pytest.fixture
def home(tmp_path: Path) -> Path:
    notes = {
        "notes/halo.md": "# Halo\n\nDark matter halo models. See [[Brush]].\n",
        "notes/brush.md": "# Brush\n\nA robot plans brush strokes.\n",
    }
    return home_of(tmp_path, notes)


def test_searches_through_what_was_decoded_find_what_they_find_alone_and_follow_the_store(
    home: Path, tmp_path: Path
) -> None:
    kept = KeptDecoded()
    watching = Watching(decoded=kept)
    asked = SearchKeywords("robot OR halo")
    alone = search(home, asked)
    assert search(home, asked, watching) == alone
    held = kept.size
    assert search(home, asked, watching) == alone
    assert (held > 0, kept.size) == (True, held)
    hybrid = SearchHybrid("robot")
    assert query(home, hybrid, Hashed(), watching=watching) == query(home, hybrid, Hashed())
    phrase = SearchKeywords('"robot plans"')
    assert search(home, phrase, watching) == search(home, phrase, watching) == search(home, phrase)
    assert mentions(home, FindMentions("notes/brush.md"), watching) == mentions(home, FindMentions("notes/brush.md"))
    write(tmp_path / "notes" / "robots.md", "# Robots\n\nA robot arm.\n")
    indexed(home)
    assert search(home, SearchKeywords("robot"), watching) == search(home, SearchKeywords("robot"))
    assert len(search(home, SearchKeywords("robot"), watching).hits) == 2


def arrays_of(value: Decoded) -> tuple[NDArray[np.generic], ...]:
    if isinstance(value, Dense):
        return value.numbers, value.table, value.order, value.blocks.values
    return value if isinstance(value, tuple) else (value,)


def test_many_small_decoded_lists_are_held_to_the_budget_with_what_each_entry_costs() -> None:
    kept = KeptDecoded(budget=100_000)
    positions = kept(A).positions
    for n in range(10_000):
        positions[1, f"w{n}", 0] = np.zeros(1, np.int64)
    assert (len(kept.entries) <= 100_000 // ENTRY_BYTES, 0 < kept.size <= kept.budget) == (True, True)


def test_what_is_kept_cannot_be_written(home: Path) -> None:
    kept = KeptDecoded()
    search(home, SearchKeywords('"robot plans" OR halo'), Watching(decoded=kept))
    arrays_kept = [array for value, _ in kept.entries.values() for array in arrays_of(value)]
    assert len(arrays_kept) == 6
    for array in arrays_kept:
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.flags.writeable = True
