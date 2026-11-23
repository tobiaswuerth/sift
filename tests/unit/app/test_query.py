import contextlib
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError
from sift.api.indexing import IndexCollections
from sift.api.search import SearchHybrid
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.app.query import query
from sift.app.vsearch import Asked
from sift.domain.arrays import Ids
from sift.domain.codec.codes import Halves
from sift.domain.errors import DamagedError
from sift.storage.database import Reading
from sift.storage.vectors import vectors_of
from tests.damage import damage_table
from tests.models import Hashed

LONG = "\n\n".join(["Pruning skips ranges that cannot win. " * 20] * 8)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def collection(home: Path, root: Path, *, embedded: bool = True) -> None:
    add(home, AddCollection(str(root)))
    index(home, IndexCollections(), lambda _: None)
    if embedded:
        embed(home, EmbedCollections((root.name,)), lambda _: None, Hashed())


@pytest.fixture
def home(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    write(root / "pruning.md", f"# Block-Max Pruning\n\nRanges are scored best bound first.\n\n## More\n\n{LONG}\n")
    write(root / "halo.md", "---\nauthor: Vera Rubin\ndoi: 10.1000/halo\n---\n# Halo\n\nDark matter halo models.\n")
    write(
        root / "matter.md",
        "---\ntags: [cosmology, physics]\nseen: {by: nobody}\n---\n# Matter\n\nDark matter, and matter again.\n",
    )
    for n in range(6):
        write(root / f"filler{n}.md", f"# Filler {n}\n\nNothing to see here, note {n}.\n")
    home = tmp_path / "home"
    collection(home, root)
    return home


def ids(home: Path, request: SearchHybrid, models: Hashed | None = None) -> list[str]:
    return [hit.id for hit in query(home, request, models or Hashed()).hits]


def test_keyword_and_semantic_candidates_are_reranked_each_document_once(home: Path) -> None:
    result = query(home, SearchHybrid("dark matter halo models", k=3), Hashed())
    assert [hit.id for hit in result.hits][:1] == ["notes/halo.md"]
    assert len({hit.id for hit in result.hits}) == len(result.hits) == 3
    assert result.hits[0].text == "# Halo\n\nDark matter halo models."
    assert [hit.score for hit in result.hits] == sorted((hit.score for hit in result.hits), reverse=True)


def test_the_reranker_reads_the_title_the_front_matter_the_query_names_and_the_text(home: Path) -> None:
    models = Hashed()
    query(home, SearchHybrid("papers by Rubin on halo models"), models)
    ((asked, shown),) = models.reranked
    assert asked == "papers by Rubin on halo models"
    assert "Halo\nVera Rubin 10.1000/halo\n\n# Halo\n\nDark matter halo models." in shown
    assert "Matter\n\n# Matter\n\nDark matter, and matter again." in shown


def test_every_passage_is_found_when_asked_and_text_left_out_when_compact(home: Path) -> None:
    assert ids(home, SearchHybrid("pruning skips ranges", k=4, all_passages=True)).count("notes/pruning.md") > 1
    assert all(hit.text is None for hit in query(home, SearchHybrid("halo", compact=True), Hashed()).hits)


def test_copies_of_a_work_in_several_collections_are_one_result_naming_the_others(home: Path, tmp_path: Path) -> None:
    write(
        tmp_path / "more" / "copy.md",
        "---\ndoi: https://doi.org/10.1000/HALO\n---\n# Halo\n\nDark matter halo models.\n",
    )
    collection(home, tmp_path / "more")
    (first, *rest) = query(home, SearchHybrid("dark matter halo models", k=5), Hashed()).hits
    assert {first.id, *first.also} == {"notes/halo.md", "more/copy.md"}
    assert len(first.also) == 1
    assert first.also[0] not in [hit.id for hit in rest]


def test_adaptive_keeps_the_results_that_stand_out(home: Path) -> None:
    every = query(home, SearchHybrid("dark matter halo models", k=9), Hashed()).hits
    standing = query(home, SearchHybrid("dark matter halo models", k=9, adaptive=True), Hashed()).hits
    assert 1 <= len(standing) <= len(every)
    assert standing == every[: len(standing)]


def test_a_collection_not_embedded_gives_keyword_candidates_alone(tmp_path: Path) -> None:
    root = tmp_path / "plain"
    write(root / "a.md", "# A\n\nComets and glaciers.\n")
    write(root / "b.md", "# B\n\nNothing here.\n")
    home = tmp_path / "home"
    collection(home, root, embedded=False)
    models = Hashed()
    result = query(home, SearchHybrid("comets"), models)
    assert [hit.id for hit in result.hits] == ["plain/a.md"]
    assert (result.unsearched, models.asked) == ((), [])


def changed(store: Path, sql: str) -> None:
    with contextlib.closing(sqlite3.connect(store)) as database, database:
        database.execute(sql)


def damaged_keywords(store: Path) -> None:
    damage_table(store, "keyword_postings")


def damaged_documents(store: Path) -> None:
    damage_table(store, "documents")


def damaged_vectors(store: Path) -> None:
    changed(store, "UPDATE vectors SET vector = x'00'")


@pytest.mark.parametrize("damaged", [damaged_keywords, damaged_documents, damaged_vectors])
def test_a_store_lost_while_its_candidates_are_read_is_passed_over(
    home: Path, tmp_path: Path, damaged: Callable[[Path], None]
) -> None:
    write(tmp_path / "more" / "halo.md", "# Halo again\n\nDark matter halo models.\n")
    collection(home, tmp_path / "more")
    damaged(home / "collections" / "notes.sqlite")
    result = query(home, SearchHybrid("dark matter halo models"), Hashed())
    assert [hit.id for hit in result.hits] == ["more/halo.md"]
    assert [each.collection for each in result.unsearched] == ["notes"]


@pytest.mark.parametrize(
    "request_, refusal",
    [
        (SearchHybrid("   "), "an empty query finds nothing"),
        (SearchHybrid("halo", k=0), "asks for no result"),
        (SearchHybrid("halo", k=20, candidates=10), "fewer than -k"),
        (SearchHybrid("halo", candidates=10_001), "more than allowed"),
    ],
)
def test_a_query_that_cannot_be_answered_is_refused(home: Path, request_: SearchHybrid, refusal: str) -> None:
    with pytest.raises(InvalidRequestError, match=refusal):
        query(home, request_, Hashed())


def test_a_store_lost_while_the_cosines_of_its_candidates_are_read_is_passed_over(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "halo.md", "# Halo again\n\nDark matter halo models.\n")
    collection(home, tmp_path / "more")
    read: list[int] = []

    def reading(transaction: Reading, chunks: Ids) -> tuple[Ids, Halves]:
        read.append(len(chunks))
        if len(read) == 3:
            raise DamagedError("a vector is unreadable")
        return vectors_of(transaction, chunks)

    result = query(home, SearchHybrid("dark matter halo models"), Hashed(), Asked(1000, reading))
    assert len(result.unsearched) == 1
    assert result.hits


def test_values_of_the_front_matter_the_query_names_are_read_from_lists_too(home: Path) -> None:
    models = Hashed()
    query(home, SearchHybrid("cosmology of matter"), models)
    ((_, shown),) = models.reranked
    assert "Matter\ncosmology\n\n# Matter\n\nDark matter, and matter again." in shown


def test_a_query_matching_nothing_finds_nothing(tmp_path: Path) -> None:
    root = tmp_path / "plain"
    write(root / "a.md", "# A\n\nComets and glaciers.\n")
    home = tmp_path / "home"
    collection(home, root, embedded=False)
    models = Hashed()
    assert query(home, SearchHybrid("volcanoes"), models).hits == ()
    assert models.reranked == []


def test_a_document_found_by_keywords_in_one_passage_and_by_meaning_in_another_is_one_result(tmp_path: Path) -> None:
    root = tmp_path / "zoo"
    sections = "".join(f"## Part {n}\n\n{'Quokka wombat numbat. ' * 60}\n\n" for n in range(10))
    write(root / "two.md", f"# Two\n\nZebra.\n\n{sections}")
    for n in range(4):
        write(root / f"filler{n}.md", f"# Filler {n}\n\nNothing to see here, note {n}.\n")
    home = tmp_path / "home"
    collection(home, root)
    models = Hashed()
    result = query(home, SearchHybrid("zebra quokka"), models)
    ((_, shown),) = models.reranked
    assert sum(each.startswith("Two\n") for each in shown) == 2
    assert [hit.id for hit in result.hits].count("zoo/two.md") == 1
