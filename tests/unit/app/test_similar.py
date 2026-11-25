import contextlib
import sqlite3
from pathlib import Path

import pytest

from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, NotFoundError, Problem, StoreUnavailableError
from sift.api.links import FindSimilar
from sift.api.search import SearchKeywords, SearchResult, SearchVectors
from sift.app.embedding import embed
from sift.app.search import search
from sift.app.similar import similar
from sift.app.vsearch import vsearch
from sift.interfaces.cli.written import written
from tests.files import write
from tests.homes import collection, indexed, quiet
from tests.models import Hashed

ROBOTS = "Robots clean floors with brushes.\n"
VACUUM = "A robot vacuum cleans floors.\n"
ZEROED = """UPDATE vectors SET vector = zeroblob(length(vector)) WHERE chunk IN (
  SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document JOIN sources s ON s.id = d.source WHERE s.path = ?)"""


@pytest.fixture
def papers(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    collection(
        tmp_path,
        home,
        "arxiv",
        {
            "robots.md": f"---\ndoi: 10.1/robots\n---\n{ROBOTS}",
            "vacuum.md": f"---\ndoi: 10.1/vacuum\n---\n{VACUUM}",
            "soup.md": "Soup with leeks and potatoes.\n",
            "bread.md": "Bread with flour and water.\n",
        },
    )
    collection(
        tmp_path,
        home,
        "openalex",
        {
            "W1.md": f'---\ndoi: "https://doi.org/10.1/ROBOTS"\n---\n{ROBOTS}',
            "W2.md": f"---\ndoi: doi:10.1/vacuum\n---\n{VACUUM}",
            "W3.md": "Robots and brushes for the floors.\n",
        },
    )
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    return home


def ids(result: SearchResult) -> list[str]:
    return [hit.id for hit in result.hits]


WORKS = ({"arxiv/robots.md", "openalex/W1.md"}, {"arxiv/vacuum.md", "openalex/W2.md"})


def one_per_work(result: SearchResult) -> None:
    for copies in WORKS:
        shown = [hit for hit in result.hits if hit.id in copies]
        assert len(shown) <= 1
        assert all(set(hit.also) <= copies - {hit.id} for hit in shown)


def named_copies(result: SearchResult) -> list[set[str]]:
    return [{hit.id, *hit.also} for hit in result.hits if hit.also]


def test_a_search_shows_one_result_per_work_and_names_its_copies(papers: Path) -> None:
    found = search(papers, SearchKeywords("robots OR clean OR floors OR brushes", k=2))
    one_per_work(found)
    assert (len(found.hits), named_copies(found)) == (2, [WORKS[0]])
    near = vsearch(papers, SearchVectors("robots clean floors", k=3), Hashed())
    one_per_work(near)
    assert (len(near.hits), named_copies(near)) == (3, [WORKS[0]])
    every = search(papers, SearchKeywords("floors", k=10, all_passages=True))
    assert len(every.hits) == 5


def test_documents_like_one_are_found_by_the_mean_of_its_vectors(papers: Path) -> None:
    found = similar(papers, FindSimilar("arxiv/robots.md", k=3))
    assert len(found.hits) == 3
    assert not {"arxiv/robots.md", "openalex/W1.md"} & set(ids(found))
    one_per_work(found)
    assert [hit.score for hit in found.hits] == sorted((hit.score for hit in found.hits), reverse=True)
    assert ids(similar(papers, FindSimilar("robots", k=1))) == ids(found)[:1]


def test_every_passage_alike_is_k_of_them_though_the_passages_of_the_copy_are_nearest(tmp_path: Path) -> None:
    home = tmp_path / "home"
    sections = "".join(f"## Part {n}\n\n{'attention transformer layers ' * 150}\n\n" for n in range(3))
    paper = f'---\ndoi: "10.1000/paper"\n---\n# Paper\n\n{sections}'
    others = {f"other{n}.md": f"# Other {n}\n\nattention transformer layers, and {n}\n" for n in range(8)}
    collection(tmp_path, home, "a", {"paper.md": paper})
    collection(tmp_path, home, "b", {"paper.md": paper, **others})
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    found = similar(home, FindSimilar("a/paper.md", k=5, all_passages=True))
    assert (len(found.hits), "b/paper.md" in ids(found)) == (5, False)


def test_similar_takes_what_searches_take(papers: Path) -> None:
    asked = FindSimilar(
        "arxiv/robots.md", collections=("openalex",), compact=True, fields=("doi",), filters=(written("doi!=10.1/x"),)
    )
    found = similar(papers, asked)
    assert sorted(ids(found)) == ["openalex/W2.md", "openalex/W3.md"]
    assert all(hit.text is None for hit in found.hits)
    assert {hit.id: hit.fields for hit in found.hits}["openalex/W2.md"] == {"doi": "doi:10.1/vacuum"}


def test_similar_needs_the_document_embedded(papers: Path, tmp_path: Path) -> None:
    write(tmp_path / "arxiv" / "new.md", "Fresh robots.\n")
    indexed(papers)
    with pytest.raises(StoreUnavailableError, match=r"'arxiv/new\.md' has no vectors yet") as refused:
        similar(papers, FindSimilar("arxiv/new.md"))
    assert (refused.value.remedy, refused.value.code) == ("embed the collection", Problem.NOT_EMBEDDED)
    assert refused.value.remedy == "embed the collection"
    with pytest.raises(NotFoundError, match="no note or document"):
        similar(papers, FindSimilar("arxiv/missing.md"))


def test_a_document_whose_vectors_point_nowhere_is_refused(papers: Path) -> None:
    store = papers / "collections" / "arxiv.sqlite"
    with contextlib.closing(sqlite3.connect(store)) as database, database:
        database.execute(ZEROED, ("soup.md",))
    with pytest.raises(InvalidRequestError, match=r"'arxiv/soup\.md' has no direction to be near to"):
        similar(papers, FindSimilar("arxiv/soup.md"))


def test_a_document_linking_the_note_is_still_alike(tmp_path: Path) -> None:
    home = tmp_path / "home"
    collection(tmp_path, home, "v", {"Robots.md": ROBOTS, "Mop.md": f"{ROBOTS}See [[Robots]].\n", "soup.md": "Soup.\n"})
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    assert ids(similar(home, FindSimilar("v/Robots.md", k=1))) == ["v/Mop.md"]
