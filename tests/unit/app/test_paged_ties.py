from collections.abc import Callable
from pathlib import Path

import pytest

from sift.api.embedding import EmbedCollections
from sift.api.search import SearchHybrid, SearchKeywords, SearchResult, SearchVectors
from sift.app.embedding import embed
from sift.app.query import query
from sift.app.search import search
from sift.app.vsearch import vsearch
from tests.files import write
from tests.homes import indexed_home
from tests.models import Hashed

COPIES = 12

type Paged = Callable[[Path, int, int], SearchResult]


def keywords(home: Path, k: int, offset: int) -> SearchResult:
    return search(home, SearchKeywords("pigment", k=k, offset=offset))


def meaning(home: Path, k: int, offset: int) -> SearchResult:
    return vsearch(home, SearchVectors("pigment", k=k, offset=offset), Hashed())


def both(home: Path, k: int, offset: int) -> SearchResult:
    return query(home, SearchHybrid("pigment", k=k, offset=offset), Hashed())


@pytest.mark.parametrize("found", [keywords, meaning, both])
def test_pages_through_hits_of_one_score_neither_overlap_nor_skip(tmp_path: Path, found: Paged) -> None:
    for n in range(COPIES):
        write(tmp_path / "copies" / f"copy{n:02}.md", "# Copy\n\nThe pigment and its binder.\n")
    home = indexed_home(tmp_path, tmp_path / "copies")
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    every = [hit.id for hit in found(home, COPIES, 0).hits]
    pages = [[hit.id for hit in found(home, 5, offset).hits] for offset in range(0, COPIES, 5)]
    assert len(set(every)) == COPIES
    assert [each for page in pages for each in page] == every
