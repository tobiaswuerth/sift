import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import override

import numpy as np
import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, ModelUnavailableError, Problem
from sift.api.events import Event, Phase, PhaseDone, PhaseStarted
from sift.api.notices import Notice
from sift.api.search import SearchHybrid
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.query import WORDS_REMEDY, query
from sift.app.searching import Watching
from sift.app.vector_candidates import Rescoring
from sift.domain.arrays import Ids, floats
from sift.domain.codec.codes import Floats, Halves
from sift.domain.errors import DamagedError
from sift.domain.query.grammar import MOST_WORDS
from sift.storage.database import Reading
from sift.storage.vectors import EMBED_REMEDY, vectors_of
from tests.damage import changed, damage_pages, damage_table
from tests.files import write
from tests.homes import home_of, indexed, long_parts, quiet
from tests.models import Hashed

LONG = "\n\n".join(["Pruning skips ranges that cannot win. " * 20] * 8)


def collection(home: Path, root: Path, *, embedded: bool = True) -> None:
    add(home, AddCollection(str(root)))
    indexed(home)
    if embedded:
        embed(home, EmbedCollections((root.name,)), quiet, Hashed())


NOTES = {
    "notes/pruning.md": f"# Block-Max Pruning\n\nRanges are scored best bound first.\n\n## More\n\n{LONG}\n",
    "notes/halo.md": "---\nauthor: Ada Quill\ndoi: 10.1000/halo\n---\n# Halo\n\nDark matter halo models.\n",
    "notes/matter.md": (
        "---\ntags: [cosmology, physics]\nseen: {by: nobody}\n---\n# Matter\n\nDark matter, and matter again.\n"
    ),
    **{f"notes/filler{n}.md": f"# Filler {n}\n\nNothing to see here, note {n}.\n" for n in range(6)},
}


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return home_of(tmp_path, NOTES, embedded=Hashed())


def ids(home: Path, request: SearchHybrid, models: Hashed | None = None) -> list[str]:
    return [hit.id for hit in query(home, request, models or Hashed()).hits]


def test_keyword_and_semantic_candidates_are_reranked_each_document_once(home: Path) -> None:
    result = query(home, SearchHybrid("dark matter halo models", k=3), Hashed())
    assert [hit.id for hit in result.hits][:1] == ["notes/halo.md"]
    assert len({hit.id for hit in result.hits}) == len(result.hits) == 3
    assert result.hits[0].text == "# Halo\n\nDark matter halo models."
    assert [hit.score for hit in result.hits] == sorted((hit.score for hit in result.hits), reverse=True)


def test_a_hit_s_score_blends_what_the_reranker_says_with_how_near_its_meaning_is(home: Path) -> None:
    result = query(home, SearchHybrid("dark matter halo models", k=4), Hashed(scored=True))
    assert [(hit.id, hit.score) for hit in result.hits] == [
        ("notes/halo.md", pytest.approx(4.130132, rel=1e-6)),
        ("notes/matter.md", pytest.approx(2.973595, rel=1e-6)),
        ("notes/filler2.md", pytest.approx(-0.1521017, rel=1e-6)),
        ("notes/filler5.md", pytest.approx(-0.1561312, rel=1e-6)),
    ]


def test_copies_of_one_work_widen_the_candidates_until_k_works_are_found(tmp_path: Path) -> None:
    home = tmp_path / "home"
    for name in ("a", "b", "c"):
        for n in range(4):
            text = f"---\ndoi: 10.1000/w{n}\n---\n# Work {n}\n\n{'quasar ' * (4 - n)}light.\n"
            write(tmp_path / name / f"w{n}.md", text)
        collection(home, tmp_path / name)
    found = query(home, SearchHybrid("quasar", k=4, candidates=4), Hashed())
    assert len(found.hits) == 4
    assert [len(hit.also) for hit in found.hits] == [2, 2, 2, 2]


def test_the_reranker_reads_the_title_the_front_matter_the_query_names_and_the_text(home: Path) -> None:
    models = Hashed()
    query(home, SearchHybrid("papers by Quill on halo models"), models)
    ((asked, shown),) = models.reranked
    assert asked == "papers by Quill on halo models"
    assert "Halo\nAda Quill 10.1000/halo\n\n# Halo\n\nDark matter halo models." in shown
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
    assert 1 <= len(standing) < len(every)
    assert standing == every[: len(standing)]


def test_a_compact_query_ranks_as_the_whole_one_without_the_text(home: Path) -> None:
    whole = query(home, SearchHybrid("dark matter halo models", k=9), Hashed()).hits
    compact = query(home, SearchHybrid("dark matter halo models", k=9, compact=True), Hashed()).hits
    assert [(hit.id, hit.lines, hit.score) for hit in compact] == [(hit.id, hit.lines, hit.score) for hit in whole]
    assert {hit.text for hit in compact} == {None}


def test_a_question_of_stopwords_alone_is_answered_by_meaning(home: Path) -> None:
    models = Hashed()
    assert len(query(home, SearchHybrid("what is the", k=3), models).hits) == 3
    assert [asked for asked, _ in models.reranked] == ["what is the"]


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
        (partial(SearchHybrid, "   "), "an empty query finds nothing"),
        (partial(SearchHybrid, "halo", k=0), "0 is outside 1 to 10,000"),
        (partial(SearchHybrid, "halo", candidates=0), "0 is outside 1 to 10,000"),
        (partial(SearchHybrid, "halo", candidates=10_001), "10,001 is outside 1 to 10,000"),
    ],
)
def test_a_query_that_cannot_be_answered_is_refused(
    home: Path, request_: Callable[[], SearchHybrid], refusal: str
) -> None:
    with pytest.raises(InvalidRequestError, match=refusal):
        query(home, request_(), Hashed())


def test_a_store_lost_while_the_cosines_of_its_candidates_are_read_is_passed_over(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "halo.md", "# Halo again\n\nDark matter halo models.\n")
    collection(home, tmp_path / "more")
    read: list[int] = []

    def reading(transaction: Reading, chunks: Ids) -> tuple[Ids, Halves]:
        read.append(len(chunks))
        if len(read) == 3:
            raise DamagedError("a vector is unreadable")
        return vectors_of(transaction, chunks)

    result = query(home, SearchHybrid("dark matter halo models"), Hashed(), Rescoring(1000, reading))
    assert len(result.unsearched) == 1
    assert [hit.id for hit in result.hits] == [
        "notes/halo.md",
        "notes/matter.md",
        "notes/filler2.md",
        "notes/filler5.md",
        "notes/filler1.md",
        "notes/pruning.md",
        "notes/filler0.md",
        "notes/filler4.md",
        "notes/filler3.md",
    ]


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


def test_fewer_candidates_than_results_are_raised_to_as_many(home: Path) -> None:
    assert len(query(home, SearchHybrid("note", k=12, candidates=2), Hashed()).hits) == 9


@pytest.mark.parametrize(
    "other, said",
    [
        (Hashed(name="other-model"), "its vectors are of other-model, not of hashed-words"),
        (Hashed(width=32), "its vectors have 32 dimensions, the query 48"),
    ],
)
def test_a_collection_embedded_otherwise_is_searched_by_keyword_alone_saying_so(
    home: Path, tmp_path: Path, other: Hashed, said: str
) -> None:
    write(tmp_path / "more" / "halo.md", "# Halo again\n\nDark matter halo models.\n")
    collection(home, tmp_path / "more", embedded=False)
    embed(home, EmbedCollections(("more",)), quiet, other)
    asked = "dark matter halo models"
    result = query(home, SearchHybrid(asked), Hashed())
    (alone,) = floats(Hashed().rerank(asked, ["Halo again\n\n# Halo again\n\nDark matter halo models."]))
    assert {hit.id: hit.score for hit in result.hits}["more/halo.md"] == alone
    assert result.unsearched == ()
    remedy = EMBED_REMEDY.format(model="hashed-words")
    assert result.left_out == (Notice(Problem.OTHER_MODEL, f"{said}: searched by keyword alone", remedy, "more"),)


def test_a_passage_not_embedded_yet_keeps_the_reranker_s_score_as_it_is(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "halo.md", "# Halo again\n\nDark matter halo models.\n")
    collection(home, tmp_path / "more", embedded=False)
    asked = "dark matter halo models"
    hits = {hit.id: hit.score for hit in query(home, SearchHybrid(asked), Hashed()).hits}
    (alone,) = floats(Hashed().rerank(asked, ["Halo again\n\n# Halo again\n\nDark matter halo models."]))
    assert hits["more/halo.md"] == alone
    assert hits["notes/halo.md"] > alone


@dataclass(frozen=True, slots=True)
class Overflowing(Hashed):
    @override
    def rerank(self, query: str, texts: Sequence[str]) -> Floats:
        return Hashed.rerank(self, query, texts) + np.float32(np.nan)


def test_a_reranker_giving_no_score_is_refused_not_ranked_at_random(home: Path) -> None:
    with pytest.raises(ModelUnavailableError, match="the reranker gave no score") as refused:
        query(home, SearchHybrid("dark matter halo models"), Overflowing())
    assert refused.value.code == Problem.MODEL_UNAVAILABLE


def test_a_question_written_decomposed_finds_what_its_words_name(tmp_path: Path) -> None:
    root = tmp_path / "plain"
    write(root / "a.md", "# Café\n\nThe résumé of Zürich.\n")
    write(root / "b.md", "# B\n\nNothing here.\n")
    home = tmp_path / "home"
    collection(home, root, embedded=False)
    result = query(home, SearchHybrid(unicodedata.normalize("NFD", "Zürich résumé")), Hashed())
    assert [hit.id for hit in result.hits] == ["plain/a.md"]


def test_scores_are_given_as_reckoned_and_rounded_only_where_shown(home: Path) -> None:
    result = query(home, SearchHybrid("dark matter halo models", k=5), Hashed())
    assert any(hit.score != round(hit.score, 4) for hit in result.hits)


def test_loading_counts_the_models_a_question_needs(home: Path, tmp_path: Path) -> None:
    said: list[Event] = []
    query(home, SearchHybrid("halo"), Hashed(), watching=Watching(report=said.append))
    assert said == [PhaseStarted(Phase.LOAD, None), PhaseDone(2)]
    root = tmp_path / "plain"
    write(root / "a.md", "# A\n\nComets and glaciers.\n")
    plain = tmp_path / "plain_home"
    collection(plain, root, embedded=False)
    words: list[Event] = []
    query(plain, SearchHybrid("comets"), Hashed(), watching=Watching(report=words.append))
    assert words == [PhaseStarted(Phase.LOAD, None), PhaseDone(1)]


def test_nothing_loads_when_no_store_is_left_to_search(home: Path) -> None:
    store = home / "collections" / "notes.sqlite"
    damage_pages(store)
    said: list[Event] = []
    result = query(home, SearchHybrid("halo"), Hashed(), watching=Watching(report=said.append))
    assert (result.hits, said) == ((), [])


def test_a_question_of_more_words_than_a_query_reads_is_answered_by_its_first_ones_saying_so(home: Path) -> None:
    question = " ".join(["halo", *(f"w{n}" for n in range(MOST_WORDS + 499))])
    result = query(home, SearchHybrid(question), Hashed())
    assert "notes/halo.md" in [hit.id for hit in result.hits]
    assert result.left_out == (
        Notice(
            Problem.TOO_MANY_WORDS,
            f"the question holds {MOST_WORDS + 500:,} words: its keywords are the first {MOST_WORDS:,}",
            WORDS_REMEDY,
        ),
    )


def test_a_query_whose_nearest_passages_belong_to_few_documents_says_it_read_no_farther(tmp_path: Path) -> None:
    home, root = tmp_path / "home", long_parts(tmp_path)
    write(root / "gamma.md", "# gamma\n\nWombats dig burrows.\n")
    collection(home, root)
    found = query(home, SearchHybrid("comets glaciers", k=3, candidates=1), Hashed(), Rescoring(1, vectors_of))
    widened = [each for each in found.left_out if each.code is Problem.WIDENED_ENOUGH]
    assert (found.partial, [each.message.endswith("a search reads no farther") for each in widened]) == (True, [True])
