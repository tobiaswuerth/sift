from pathlib import Path

import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.indexing import IndexCollections
from sift.api.links import Direction, FindSimilar, ListUnresolved, ShowLinks
from sift.api.search import SearchHybrid, SearchKeywords, SearchVectors
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.app.links import links, unresolved
from sift.app.query import query
from sift.app.search import search
from sift.app.similar import similar
from sift.app.vsearch import vsearch
from tests.files import write
from tests.models import Hashed

NOTES = 9


@pytest.fixture
def home(tmp_path: Path) -> Path:
    root, home = tmp_path / "notes", tmp_path / "home"
    for n in range(NOTES):
        after = (n + 1) % NOTES
        write(root / f"halo{n}.md", f"# Halo {n}\n\nDark matter halo {n}: see [[Missing {n}]] and [[halo{after}]].\n")
    add(home, AddCollection(str(root)))
    index(home, IndexCollections(), lambda _: None)
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    return home


PAGES = [(2, 0, True), (8, 0, True), (9, 0, False), (10, 0, False), (4, 4, True), (4, 5, False), (1, 8, False)]


@pytest.mark.parametrize("k, offset, more", PAGES)
def test_a_search_says_whether_more_results_follow_those_it_gives(
    home: Path, k: int, offset: int, *, more: bool
) -> None:
    for found in (
        search(home, SearchKeywords("halo", k=k, offset=offset)),
        vsearch(home, SearchVectors("dark matter halo", k=k, offset=offset), Hashed()),
        query(home, SearchHybrid("dark matter halo", k=k, offset=offset), Hashed()),
    ):
        assert (len(found.hits), found.more) == (max(0, min(k, NOTES - offset)), more)


@pytest.mark.parametrize("k, more", [(7, True), (8, False), (9, False)])
def test_a_search_for_notes_like_one_says_whether_more_follow(home: Path, k: int, *, more: bool) -> None:
    assert similar(home, FindSimilar("notes/halo0.md", k=k)).more is more


def test_an_adaptive_search_has_more_where_it_kept_fewer_than_it_found(home: Path) -> None:
    found = search(home, SearchKeywords("halo", k=NOTES, adaptive=True))
    assert found.more is (len(found.hits) < NOTES)


def test_a_search_says_when_no_passage_held_every_word_and_any_was_searched_for(home: Path) -> None:
    loosened = search(home, SearchKeywords("halo zebra"))
    assert (len(loosened.hits), loosened.loosened) == (NOTES, True)
    assert search(home, SearchKeywords("halo matter")).loosened is False
    assert search(home, SearchKeywords("zebra quagga")).loosened is False


def test_a_search_says_what_its_scores_measure(home: Path) -> None:
    assert search(home, SearchKeywords("halo")).scored_by == "keywords"
    assert vsearch(home, SearchVectors("halo"), Hashed()).scored_by == "vectors"
    assert similar(home, FindSimilar("notes/halo0.md")).scored_by == "vectors"
    assert query(home, SearchHybrid("halo"), Hashed()).scored_by == "reranker"


@pytest.mark.parametrize("k, more", [(1, True), (2, False)])
def test_links_say_whether_more_follow_those_given(home: Path, k: int, *, more: bool) -> None:
    assert links(home, ShowLinks("notes/halo0.md", Direction.OUT, k=k)).more is more
    assert links(home, ShowLinks("notes/halo0.md", Direction.BOTH, depth=2, k=k)).more is True


@pytest.mark.parametrize("k, more", [(5, True), (8, True), (9, False)])
def test_unresolved_targets_say_whether_more_follow_those_given(home: Path, k: int, *, more: bool) -> None:
    assert unresolved(home, ListUnresolved(k=k)).more is more
