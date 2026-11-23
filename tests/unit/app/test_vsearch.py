import contextlib
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, NotFoundError
from sift.api.events import Event, Phase, PhaseDone, PhaseStarted
from sift.api.indexing import IndexCollections
from sift.api.search import SearchVectors, Unsearched
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.app.searching import UNEMBEDDED_REMEDY
from sift.app.vector_candidates import Rescoring
from sift.app.vsearch import vsearch
from sift.domain.errors import DamagedError
from sift.storage.database import Reading
from sift.storage.vectors import EMBED_REMEDY, documents_embedded, vectors_of
from tests.counting import DocumentCounts, VectorReads
from tests.damage import changed, damage_table
from tests.files import write
from tests.models import BROKEN, DIRECTIONLESS, Hashed

LONG = "\n\n".join(["Pruning skips ranges that cannot win. " * 20] * 8)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    write(root / "pruning.md", f"# Block-Max Pruning\n\nRanges are scored best bound first.\n\n## More\n\n{LONG}\n")
    write(root / "halo.md", "# Halo\n\nDark matter halo models.\n")
    write(root / "matter.md", "# Matter\n\nDark matter, and matter again.\n")
    for n in range(6):
        write(root / f"filler{n}.md", f"# Filler {n}\n\nNothing to see here, note {n}.\n")
    home = tmp_path / "home"
    add(home, AddCollection(str(root)))
    index(home, IndexCollections(), lambda _: None)
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    return home


def more(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "halo.md", "# Halo again\n\nDark matter halo models.\n")
    add(home, AddCollection(str(tmp_path / "more")))
    index(home, IndexCollections(), lambda _: None)


def ids(home: Path, request: SearchVectors) -> list[str]:
    return [hit.id for hit in vsearch(home, request, Hashed()).hits]


def test_passages_nearest_in_meaning_are_found_each_document_once(home: Path) -> None:
    result = vsearch(home, SearchVectors("dark matter halo models", k=3), Hashed())
    assert [hit.id for hit in result.hits][:1] == ["notes/halo.md"]
    assert len({hit.id for hit in result.hits}) == len(result.hits) == 3
    assert result.hits[0].text == "# Halo\n\nDark matter halo models."
    assert result.hits[0].lines == "1-3"
    assert result.searched == 1
    scores = [hit.score for hit in result.hits]
    assert scores == sorted(scores, reverse=True)


def test_every_passage_is_found_when_asked_and_text_left_out_when_compact(home: Path) -> None:
    found = ids(home, SearchVectors("pruning skips ranges that cannot win", k=4, all_passages=True))
    assert found.count("notes/pruning.md") > 1
    compact = vsearch(home, SearchVectors("halo", compact=True), Hashed())
    assert all(hit.text is None for hit in compact.hits)


def test_the_query_is_embedded_as_a_query(home: Path) -> None:
    models = Hashed()
    vsearch(home, SearchVectors("halo"), models)
    assert models.asked == [(("halo",), True)]


@pytest.mark.parametrize(
    "request_, refusal",
    [
        (SearchVectors("   "), "an empty query finds nothing"),
        (SearchVectors("halo", k=0), "0 is outside 1 to 10,000"),
        (SearchVectors("halo", k=10_001), "10,001 is outside 1 to 10,000"),
        (SearchVectors(f"a {BROKEN} query"), "the model gave no vector for the query"),
        (SearchVectors(f"a {DIRECTIONLESS} query"), "the model gave the query no direction"),
    ],
)
def test_a_query_that_cannot_be_searched_is_refused(home: Path, request_: SearchVectors, refusal: str) -> None:
    with pytest.raises(InvalidRequestError, match=refusal):
        vsearch(home, request_, Hashed())


def test_a_collection_not_yet_embedded_is_passed_over_saying_so(home: Path, tmp_path: Path) -> None:
    more(home, tmp_path)
    result = vsearch(home, SearchVectors("halo"), Hashed())
    assert result.unsearched == (Unsearched("more", "it holds no vectors yet", UNEMBEDDED_REMEDY),)
    assert result.searched == 1


def test_a_collection_of_another_model_is_passed_over_saying_so(home: Path, tmp_path: Path) -> None:
    more(home, tmp_path)
    embed(home, EmbedCollections(("more",)), lambda _: None, Hashed(name="other-model"))
    result = vsearch(home, SearchVectors("halo"), Hashed())
    assert result.unsearched == (
        Unsearched(
            "more", "its vectors are of other-model, not of hashed-words", EMBED_REMEDY.format(model="hashed-words")
        ),
    )


def test_collections_are_searched_together_and_one_alone_when_named(home: Path, tmp_path: Path) -> None:
    more(home, tmp_path)
    embed(home, EmbedCollections(("more",)), lambda _: None, Hashed())
    together = ids(home, SearchVectors("dark matter halo models", k=2))
    assert set(together) == {"notes/halo.md", "more/halo.md"}
    assert ids(home, SearchVectors("dark matter halo models", k=2, collections=("more",))) == ["more/halo.md"]


def test_a_store_whose_vectors_are_damaged_is_passed_over_and_the_rest_searched(home: Path, tmp_path: Path) -> None:
    more(home, tmp_path)
    embed(home, EmbedCollections(("more",)), lambda _: None, Hashed())
    changed(home / "collections" / "notes.sqlite", "UPDATE vectors SET vector = x'00'")
    result = vsearch(home, SearchVectors("dark matter halo models"), Hashed())
    assert [hit.id for hit in result.hits] == ["more/halo.md"]
    assert [each.collection for each in result.unsearched] == ["notes"]


@pytest.mark.parametrize("table", ["chunks", "sources"])
def test_a_store_lost_while_its_hits_are_read_is_passed_over(home: Path, tmp_path: Path, table: str) -> None:
    more(home, tmp_path)
    embed(home, EmbedCollections(("more",)), lambda _: None, Hashed())
    alone = vsearch(home, SearchVectors("dark matter halo models", collections=("more",)), Hashed()).hits
    damage_table(home / "collections" / "notes.sqlite", table)
    result = vsearch(home, SearchVectors("dark matter halo models"), Hashed())
    assert result.hits == alone
    assert [each.collection for each in result.unsearched] == ["notes"]


def test_the_query_is_not_embedded_when_nothing_could_be_searched(home: Path, tmp_path: Path) -> None:
    models = Hashed()
    with pytest.raises(NotFoundError):
        vsearch(home, SearchVectors("halo", collections=("nowhere",)), models)
    more(home, tmp_path)
    assert vsearch(home, SearchVectors("halo", collections=("more",)), models).hits == ()
    assert models.asked == []


def test_a_collection_without_passages_is_searched_as_empty(home: Path, tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    add(home, AddCollection(str(tmp_path / "empty")))
    index(home, IndexCollections(), lambda _: None)
    result = vsearch(home, SearchVectors("halo"), Hashed())
    assert (result.searched, result.unsearched) == (2, ())


def test_documents_of_many_passages_widen_the_search_until_k_documents_are_found(home: Path) -> None:
    found = vsearch(
        home, SearchVectors("pruning skips ranges that cannot win", k=3), Hashed(), Rescoring(1, vectors_of)
    )
    assert len({hit.id for hit in found.hits}) == len(found.hits) == 3


def test_a_collection_of_the_same_model_at_another_width_is_passed_over_saying_so(home: Path) -> None:
    result = vsearch(home, SearchVectors("halo"), Hashed(width=32))
    assert result.unsearched == (
        Unsearched("notes", "its vectors have 48 dimensions, the query 32", EMBED_REMEDY.format(model="hashed-words")),
    )


def test_a_store_lost_while_rescoring_leaves_the_nearest_of_the_rest(home: Path, tmp_path: Path) -> None:
    more(home, tmp_path)
    embed(home, EmbedCollections(("more",)), lambda _: None, Hashed())
    changed(home / "collections" / "notes.sqlite", "UPDATE vectors SET vector = x'00'")
    found = vsearch(home, SearchVectors("# halo dark matter halo models.", k=1), Hashed(), Rescoring(1, vectors_of))
    assert [hit.id for hit in found.hits] == ["more/halo.md"]


def parts(tmp_path: Path) -> Path:
    root = tmp_path / "parts"
    for name, word in (("alpha", "comets"), ("beta", "glaciers")):
        sections = "".join(f"## Part {n}\n\n{f'About {word} and more {word}. ' * 120}\n\n" for n in range(20))
        write(root / f"{name}.md", f"# {name}\n\n{sections}")
    return root


def test_once_every_document_is_found_no_more_vectors_are_read(tmp_path: Path) -> None:
    root = parts(tmp_path)
    home = tmp_path / "home"
    add(home, AddCollection(str(root)))
    index(home, IndexCollections(), lambda _: None)
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    reads = VectorReads()
    found = vsearch(home, SearchVectors("comets glaciers", k=3), Hashed(), Rescoring(1, reads))
    assert sorted(hit.id for hit in found.hits) == ["parts/alpha.md", "parts/beta.md"]
    assert 0 < max(reads.read) < chunks_in(home / "collections" / "parts.sqlite")


def chunks_in(store: Path) -> int:
    with contextlib.closing(sqlite3.connect(store)) as database:
        (found,) = cast("tuple[int]", database.execute("SELECT count(*) FROM chunks").fetchone())
        return found


@pytest.mark.parametrize("third", ["", f"# C\n\nAn {BROKEN} note.\n", "# C\n\nIndexed, not embedded yet.\n"])
def test_documents_without_vectors_do_not_keep_the_search_widening(tmp_path: Path, third: str) -> None:
    root = parts(tmp_path)
    home = tmp_path / "home"
    add(home, AddCollection(str(root)))
    if third != "# C\n\nIndexed, not embedded yet.\n":
        write(root / "c.md", third)
    index(home, IndexCollections(), lambda _: None)
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    write(root / "c.md", third)
    index(home, IndexCollections(), lambda _: None)
    reads = VectorReads()
    found = vsearch(home, SearchVectors("comets glaciers", k=3), Hashed(), Rescoring(1, reads))
    assert sorted(hit.id for hit in found.hits) == ["parts/alpha.md", "parts/beta.md"]
    assert 0 < max(reads.read) < chunks_in(home / "collections" / "parts.sqlite") - 1


def test_documents_are_counted_only_when_the_first_round_finds_too_few(home: Path) -> None:
    count = DocumentCounts()
    vsearch(home, SearchVectors("dark matter halo models", k=3), Hashed(), Rescoring(1000, vectors_of, count))
    assert count.counted == []
    vsearch(home, SearchVectors("pruning skips ranges that cannot win", k=3), Hashed(), Rescoring(1, vectors_of, count))
    assert count.counted == [9]


def test_a_first_round_holding_every_embedded_document_is_not_widened(tmp_path: Path) -> None:
    root, home = parts(tmp_path), tmp_path / "home"
    add(home, AddCollection(str(root)))
    index(home, IndexCollections(), lambda _: None)
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    rescore = chunks_in(home / "collections" / "parts.sqlite") - 1
    reads, count = VectorReads(), DocumentCounts()
    found = vsearch(home, SearchVectors("comets glaciers", k=3), Hashed(), Rescoring(rescore, reads, count))
    assert sorted(hit.id for hit in found.hits) == ["parts/alpha.md", "parts/beta.md"]
    assert (reads.read, count.counted) == ([rescore], [2])


def test_a_store_lost_while_its_documents_are_counted_is_passed_over(home: Path, tmp_path: Path) -> None:
    more(home, tmp_path)
    embed(home, EmbedCollections(("more",)), lambda _: None, Hashed())
    counted: list[int] = []

    def count(transaction: Reading) -> int:
        counted.append(1)
        if len(counted) == 1:
            raise DamagedError("the count of embedded documents is unreadable")
        return documents_embedded(transaction)

    request = SearchVectors("pruning skips ranges that cannot win", k=3)
    found = vsearch(home, request, Hashed(), Rescoring(1, vectors_of, count))
    assert len(found.unsearched) == 1
    assert found.hits


def test_documents_are_counted_once_however_often_the_search_widens(tmp_path: Path) -> None:
    root, home = tmp_path / "zoo", tmp_path / "home"
    write(
        root / "long.md",
        "# Long\n\n" + "".join(f"## Part {n}\n\n{'Quokka wombat numbat. ' * 60}\n\n" for n in range(40)),
    )
    for n in range(5):
        write(root / f"filler{n}.md", f"# Filler {n}\n\nNothing {n}.\n")
    add(home, AddCollection(str(root)))
    index(home, IndexCollections(), lambda _: None)
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    count = DocumentCounts()
    found = vsearch(home, SearchVectors("quokka wombat", k=3), Hashed(), Rescoring(1, vectors_of, count))
    assert (len(found.hits), count.counted) == (3, [6])


def test_the_model_s_loading_is_said_only_when_a_vector_is_needed(home: Path, tmp_path: Path) -> None:
    said: list[Event] = []
    vsearch(home, SearchVectors("halo"), Hashed(), report=said.append)
    assert said == [PhaseStarted(Phase.LOAD, None), PhaseDone(1)]
    bare = tmp_path / "bare"
    write(tmp_path / "plain" / "a.md", "# A\n\nComets.\n")
    add(bare, AddCollection(str(tmp_path / "plain")))
    index(bare, IndexCollections(), lambda _: None)
    quiet: list[Event] = []
    vsearch(bare, SearchVectors("halo"), Hashed(), report=quiet.append)
    assert quiet == []
