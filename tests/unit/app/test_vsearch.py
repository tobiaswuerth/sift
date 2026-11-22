import contextlib
import sqlite3
from pathlib import Path

import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError
from sift.api.indexing import IndexCollections
from sift.api.search import SearchVectors, Unsearched
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.app.vsearch import UNEMBEDDED_REMEDY, vsearch
from sift.storage.vectors import EMBED_REMEDY
from tests.damage import damage_table
from tests.models import BROKEN, Hashed

LONG = "\n\n".join(["Pruning skips ranges that cannot win. " * 20] * 8)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def changed(store: Path, sql: str) -> None:
    with contextlib.closing(sqlite3.connect(store)) as database, database:
        database.execute(sql)


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
        (SearchVectors("halo", k=0), "asks for no result"),
        (SearchVectors("halo", k=10_001), "more than 10,000 results"),
        (SearchVectors(f"a {BROKEN} query"), "the model gave no vector for the query"),
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
    assert ids(home, SearchVectors("dark matter halo models", k=2, collection="more")) == ["more/halo.md"]


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
    alone = vsearch(home, SearchVectors("dark matter halo models", collection="more"), Hashed()).hits
    damage_table(home / "collections" / "notes.sqlite", table)
    result = vsearch(home, SearchVectors("dark matter halo models"), Hashed())
    assert result.hits == alone
    assert [each.collection for each in result.unsearched] == ["notes"]
