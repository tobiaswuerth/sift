import dataclasses
import threading
from pathlib import Path

import numpy as np
import pytest

from sift.api.collections import AddCollection
from sift.api.ids import CollectionId
from sift.api.indexing import IndexCollections
from sift.api.links import FindMentions
from sift.api.search import SearchHybrid, SearchKeywords
from sift.app.collections import add
from sift.app.indexing import index
from sift.app.kept_decoded import KeptDecoded
from sift.app.mentions import mentions
from sift.app.query import query
from sift.app.resident import KeywordsAt
from sift.app.search import search
from sift.app.searching import Watching
from sift.retrieval.keyword.lists import PostingArrays
from tests.files import write
from tests.models import Hashed

A = KeywordsAt(Path("a.sqlite"), CollectionId(2), 1)
B = KeywordsAt(Path("b.sqlite"), CollectionId(3), 1)


def arrays(n: int) -> PostingArrays:
    return np.arange(n, dtype=np.int64), np.ones(n, np.int64)


def test_decoded_blocks_are_kept_to_a_budget_of_bytes_the_longest_unused_let_go_first() -> None:
    kept = KeptDecoded(budget=1000)
    blocks = kept(A).blocks
    for term in ("a", "b", "c"):
        blocks[1, term, 0] = arrays(20)
    assert blocks.get((1, "a", 0)) is not None
    blocks[1, "d", 0] = arrays(20)
    held = [term for term in "abcd" if blocks.get((1, term, 0)) is not None]
    assert (held, kept.size) == (["a", "c", "d"], 960)


def test_a_block_larger_than_the_budget_is_not_kept() -> None:
    kept = KeptDecoded(budget=100)
    blocks = kept(A).blocks
    blocks[1, "small", 0] = arrays(2)
    blocks[1, "large", 0] = arrays(20)
    assert (blocks.get((1, "large", 0)), kept.size) == (None, 32)


def test_what_a_store_held_at_another_generation_is_let_go_and_other_stores_keep_theirs() -> None:
    kept = KeptDecoded(budget=1 << 20)
    kept(A).blocks[1, "x", 0] = arrays(4)
    kept(B).blocks[1, "x", 0] = arrays(4)
    later = kept(dataclasses.replace(A, generation=2)).blocks
    assert (later.get((1, "x", 0)), kept(B).blocks.get((1, "x", 0)) is not None, kept.size) == (None, True, 64)
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
        thread.join(60)
    assert 0 < kept.size <= 4000
    assert kept.size == sum(size for _, size in kept.entries.values())


@pytest.fixture
def home(tmp_path: Path) -> Path:
    write(tmp_path / "notes" / "halo.md", "# Halo\n\nDark matter halo models. See [[Brush]].\n")
    write(tmp_path / "notes" / "brush.md", "# Brush\n\nA robot plans brush strokes.\n")
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "notes")))
    index(home, IndexCollections(), lambda _: None)
    return home


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
    index(home, IndexCollections(), lambda _: None)
    assert search(home, SearchKeywords("robot"), watching) == search(home, SearchKeywords("robot"))
    assert len(search(home, SearchKeywords("robot"), watching).hits) == 2
