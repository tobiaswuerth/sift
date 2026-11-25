import itertools
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import pytest

from sift.api.collections import AddCollection, Audience
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, Problem, StoreUnavailableError
from sift.api.filters import Filter, Operator
from sift.api.links import FindSimilar
from sift.api.search import SORT, SearchHybrid, SearchKeywords, SearchResult, SearchVectors
from sift.app import searching
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.query import query
from sift.app.search import search
from sift.app.searching import Watching
from sift.app.similar import similar
from sift.app.vector_candidates import ExactRows, Rescoring
from sift.app.vsearch import vsearch
from sift.domain.arrays import Ids
from sift.domain.rank.hybrid import standing_out
from sift.interfaces.cli.written import listed, written
from sift.storage.database import Reading
from sift.storage.vectors import vectors_of
from tests.counting import DocumentCounts
from tests.damage import damage_table
from tests.files import write
from tests.homes import home_of, indexed, quiet
from tests.models import Hashed

DAYS = ("2023-11-30", "2024-02-10", "2024-06-15", "2024-12-31", "2025-03-01")
ANY, NONE = Operator.ANY, Operator.NONE
NOTES = {
    **{
        f"notes/dated/{day}.md": (
            f"---\ncreated: {day}\nrank: {n}\ntags: [paint, {'oil' if n % 2 else 'water'}]\n---\n"
            f"# Pigment {day}\n\nPigment and binder, part {n}.\n"
        )
        for n, day in enumerate(DAYS)
    },
    "notes/loose/pigment.md": "---\nrank: high\n---\n# Pigment notes\n\nPigment pigment pigment.\n",
    "notes/loose/plain.md": "# Plain\n\nPigment without front matter.\n",
}


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return home_of(tmp_path, NOTES, embedded=Hashed())


def days(result: SearchResult) -> list[str]:
    return sorted(hit.id.rsplit("/", 1)[1].removesuffix(".md") for hit in result.hits)


Search = Callable[[Path, tuple[Filter, ...]], SearchResult]


def by_keywords(home: Path, filters: tuple[Filter, ...]) -> SearchResult:
    return search(home, SearchKeywords("pigment binder", k=20, filters=filters))


def by_meaning(home: Path, filters: tuple[Filter, ...]) -> SearchResult:
    return vsearch(home, SearchVectors("pigment binder", k=20, filters=filters), Hashed())


def by_both(home: Path, filters: tuple[Filter, ...]) -> SearchResult:
    return query(home, SearchHybrid("pigment binder", k=20, filters=filters), Hashed())


@pytest.mark.parametrize("found", [by_keywords, by_meaning, by_both])
def test_every_search_keeps_to_the_documents_the_filters_let_through(home: Path, found: Search) -> None:
    within = (Filter("created", Operator.AT_LEAST, ("2024-06",)), Filter("created", Operator.BELOW, ("2025",)))
    assert days(found(home, within)) == ["2024-06-15", "2024-12-31"]
    assert days(found(home, (Filter("tags", ANY, ("oil",)),))) == ["2024-02-10", "2024-12-31"]
    assert days(found(home, (Filter("tags", NONE, ("paint",)),))) == ["pigment", "plain"]


Adapted = tuple[SearchResult, SearchResult]


def keywords_of(home: Path) -> Adapted:
    asked = SearchKeywords("pigment", k=20)
    return search(home, asked), search(home, replace(asked, adaptive=True))


def meaning_of(home: Path) -> Adapted:
    asked = SearchVectors("pigment", k=20)
    return vsearch(home, asked, Hashed()), vsearch(home, replace(asked, adaptive=True), Hashed())


def both_of(home: Path) -> Adapted:
    asked = SearchHybrid("pigment", k=20)
    return query(home, asked, Hashed()), query(home, replace(asked, adaptive=True), Hashed())


def like_of(home: Path) -> Adapted:
    asked = FindSimilar("notes/loose/plain.md", k=20)
    return similar(home, asked), similar(home, replace(asked, adaptive=True))


Paged = Callable[[Path, int, int], SearchResult]


def keywords_paged(home: Path, k: int, offset: int) -> SearchResult:
    return search(home, SearchKeywords("pigment binder", k=k, offset=offset))


def meaning_paged(home: Path, k: int, offset: int) -> SearchResult:
    return vsearch(home, SearchVectors("pigment binder", k=k, offset=offset), Hashed())


def both_paged(home: Path, k: int, offset: int) -> SearchResult:
    return query(home, SearchHybrid("pigment binder", k=k, offset=offset), Hashed())


def like_paged(home: Path, k: int, offset: int) -> SearchResult:
    return similar(home, FindSimilar("notes/loose/plain.md", k=k, offset=offset))


@pytest.mark.parametrize("found", [keywords_paged, meaning_paged, both_paged, like_paged])
def test_every_search_pages_through_what_it_finds_as_one_search_of_them_all_would_order_it(
    home: Path, found: Paged
) -> None:
    every = found(home, 20, 0).hits
    assert len(every) >= 4
    pages = [found(home, 2, offset).hits for offset in range(0, len(every), 2)]
    assert [hit for page in pages for hit in page] == list(every)
    assert found(home, 2, len(every)).hits == ()


@pytest.mark.parametrize(
    "asked, said",
    [
        (SearchKeywords("pigment", k=11, offset=9_990), "10,001 results"),
        (SearchKeywords("pigment", offset=2, sort="created"), "a sorted search orders the best k alone"),
        (SearchKeywords("pigment", offset=2, adaptive=True), "an adaptive search keeps what stands out"),
    ],
)
def test_a_page_no_search_of_all_would_hold_is_refused(home: Path, asked: SearchKeywords, said: str) -> None:
    with pytest.raises(InvalidRequestError, match=said) as refused:
        search(home, asked)
    assert refused.value.field == "offset"


@pytest.mark.parametrize("found", [keywords_of, meaning_of, both_of, like_of])
def test_every_search_asked_to_adapt_keeps_the_results_that_stand_out_of_all_it_finds(
    home: Path, found: Callable[[Path], Adapted]
) -> None:
    whole, adapted = found(home)
    assert adapted.hits == whole.hits[: standing_out([hit.score for hit in whole.hits])]


def test_a_keyword_search_asked_to_adapt_leaves_out_what_falls_behind_the_largest_drop(home: Path) -> None:
    whole, adapted = keywords_of(home)
    assert len(adapted.hits) < len(whole.hits)


def test_a_filter_leaves_as_many_results_as_asked_for_when_better_matches_fail_it(home: Path) -> None:
    oil = (Filter("tags", ANY, ("oil",)),)
    assert len(search(home, SearchKeywords("pigment", k=2, filters=oil)).hits) == 2
    assert len(vsearch(home, SearchVectors("pigment", k=2, filters=oil), Hashed()).hits) == 2


def test_a_filter_of_twenty_thousand_values_is_answered(home: Path) -> None:
    many = Filter("rank", ANY, (*range(10, 20_010), 2))
    assert days(search(home, SearchKeywords("pigment", k=20, filters=(many,)))) == ["2024-06-15"]


def test_a_document_without_front_matter_has_no_fields_however_many_are_asked(home: Path) -> None:
    found = search(home, SearchKeywords("pigment", k=20, fields=("*",))).hits
    assert {hit.id: hit.fields for hit in found}["notes/loose/plain.md"] == {}


def test_a_field_named_in_letters_beyond_ascii_is_filtered_and_shown_by_its_name(tmp_path: Path) -> None:
    for name, street in (("a", "Hauptweg"), ("b", "Nebenweg")):
        write(tmp_path / "streets" / f"{name}.md", f"---\nStraße: {street}\n---\n# {name}\n\nPigment.\n")
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "streets")))
    indexed(home)
    asked = SearchKeywords("pigment", filters=(written("Straße=Hauptweg"),), fields=("Straße",))
    assert [(hit.id, hit.fields) for hit in search(home, asked).hits] == [("streets/a.md", {"Straße": "Hauptweg"})]


def test_a_path_glob_keeps_to_the_documents_whose_path_matches_it(home: Path) -> None:
    assert days(search(home, SearchKeywords("pigment", k=20, path="loose/*"))) == ["pigment", "plain"]


def test_words_no_passage_holds_together_find_any_of_them_within_the_filters(home: Path) -> None:
    found = search(home, SearchKeywords("binder nowhere", k=20, filters=(Filter("rank", ANY, (1,)),)))
    assert days(found) == ["2024-02-10"]


def test_the_front_matter_asked_for_comes_with_each_hit(home: Path) -> None:
    (hit,) = search(home, SearchKeywords("pigment", path="dated/2024-02-10.md", fields=("rank", "missing"))).hits
    assert hit.fields == {"rank": 1}
    (every,) = search(home, SearchKeywords("pigment", path="dated/2024-02-10.md", fields=("*",))).hits
    assert every.fields == {"created": "2024-02-10", "rank": 1, "tags": ["paint", "oil"]}
    assert search(home, SearchKeywords("pigment", k=1)).hits[0].fields is None


def test_the_k_found_are_ordered_by_a_field_numbers_first_and_those_without_it_last(home: Path) -> None:
    ranked = search(home, SearchKeywords("pigment", k=20, sort="rank")).hits
    assert [hit.id.rsplit("/", 1)[1] for hit in ranked] == [f"{day}.md" for day in DAYS] + ["pigment.md", "plain.md"]
    newest = vsearch(home, SearchVectors("pigment", k=20, sort="-created"), Hashed()).hits
    assert [hit.id.rsplit("/", 1)[1] for hit in newest][:2] == ["2025-03-01.md", "2024-12-31.md"]
    assert newest[-1].fields is None


def test_the_sort_orders_the_k_best_not_every_match(home: Path) -> None:
    best = search(home, SearchKeywords("pigment", k=2)).hits
    sorted_best = search(home, SearchKeywords("pigment", k=2, sort="-created")).hits
    assert {hit.id for hit in sorted_best} == {hit.id for hit in best}


@pytest.mark.parametrize(
    "request_, refusal",
    [
        (SearchKeywords("pigment", sort="-"), "names no field to sort by"),
    ],
)
def test_options_that_cannot_be_followed_are_refused_saying_how(
    home: Path, request_: SearchKeywords, refusal: str
) -> None:
    with pytest.raises(InvalidRequestError, match=refusal) as refused:
        search(home, request_)
    assert refused.value.remedy == "name a front matter field, e.g. created, or -created for the newest first"


def test_a_filter_letting_few_documents_through_ends_the_semantic_search_once_they_are_found(home: Path) -> None:
    count = DocumentCounts()
    one = (Filter("rank", ANY, (1,)),)
    found = vsearch(home, SearchVectors("pigment", k=5, filters=one), Hashed(), Rescoring(1, vectors_of, count))
    assert (days(found), count.counted) == (["2024-02-10"], [])


def test_no_fields_written_asks_for_none() -> None:
    assert listed("") == ()
    assert listed(" url, created ") == ("url", "created")
    assert "-created for the newest first" in SORT


def many(tmp_path: Path, name: str, notes: int) -> Path:
    root = tmp_path / name
    for n in range(notes):
        write(root / ("even" if n % 2 == 0 else "odd") / f"{n}.md", f"# Note {n}\n\nPigment {n} and binder.\n")
    return root


@pytest.mark.parametrize(
    "notes, left",
    [
        (400, ["second/even/42.md", "second/even/168.md", "second/even/214.md"]),
        (6, ["second/even/2.md", "second/even/0.md", "second/even/4.md"]),
    ],
)
def test_a_store_lost_while_its_filter_is_checked_or_listed_is_passed_over(
    tmp_path: Path, notes: int, left: list[str]
) -> None:
    home = tmp_path / "home"
    for name in ("first", "second"):
        add(home, AddCollection(str(many(tmp_path, name, notes))))
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    request = SearchVectors("pigment binder", k=3, path="even/*")
    kept = vsearch(home, request, Hashed(), Rescoring(3, vectors_of))
    assert len(kept.hits) == 3
    assert all("/even/" in hit.id for hit in kept.hits)
    damage_table(home / "collections" / "first.sqlite", "documents")
    found = vsearch(home, request, Hashed(), Rescoring(3, vectors_of))
    assert [each.collection for each in found.unsearched] == ["first"]
    assert [hit.id for hit in found.hits] == left


def _gone() -> None:
    raise StoreUnavailableError("gone", "add it again")


def test_work_that_loses_its_store_drops_it_says_why_and_ends_the_round(home: Path) -> None:
    narrowing = searching.narrowed(home, SearchKeywords("x"), Audience.PEOPLE)
    with ExitStack() as cleanup:
        kept = searching.opened(cleanup, narrowing, searching.Wants())
        (each,) = kept.searching
        with pytest.raises(searching.StoreLostError):
            kept.guarded(each, _gone)
        assert kept.searching == []
        assert [(one.collection, one.message) for one in kept.unsearched] == [(each.collection.name, "gone")]


def test_a_round_runs_again_after_each_store_it_loses() -> None:
    losses = [searching.StoreLostError("first"), searching.StoreLostError("second")]

    def work() -> int:
        if losses:
            raise losses.pop()
        return 3

    assert searching.despite_losses(work) == 3
    assert losses == []


def test_a_filter_keeps_a_document_of_many_passages_whose_ids_are_not_its_passages_ids(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    for n in range(5):
        write(root / f"a{n}.md", f"# Filler {n}\n\nPigment {n}.\n")
    sections = "".join(f"## Part {n}\n\n{'Pigment and binder. ' * 40}\n\n" for n in range(12))
    write(root / "z.md", f"---\nrank: 9\n---\n# Long\n\n{sections}")
    home = tmp_path / "home"
    add(home, AddCollection(str(root)))
    indexed(home)
    nine = (Filter("rank", ANY, (9,)),)
    found = search(home, SearchKeywords("pigment", k=3, all_passages=True, filters=nine))
    assert [hit.id for hit in found.hits] == ["notes/z.md"] * 3


def test_once_a_store_is_lost_no_other_store_is_read_for_that_round(tmp_path: Path) -> None:
    home = tmp_path / "home"
    for name in ("first", "second"):
        add(home, AddCollection(str(many(tmp_path, name, 6))))
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    reads: list[int] = []

    def read(transaction: Reading, chunks: Ids) -> ExactRows:
        reads.append(len(chunks))
        if len(reads) == 1:
            raise StoreUnavailableError("the store is damaged", "add it again")
        return vectors_of(transaction, chunks)

    found = vsearch(home, SearchVectors("pigment binder", k=3), Hashed(), Rescoring(3, read))
    assert [each.collection for each in found.unsearched] == ["first"]
    assert len(reads) == 2


def test_a_filter_matches_text_however_its_accents_were_composed(tmp_path: Path) -> None:
    root, home = tmp_path / "cafes", tmp_path / "cafes-home"
    write(root / "decomposed.md", "---\nplace: Ro\u0308sterei\n---\n# Beans\n\nRoasted beans.\n")
    add(home, AddCollection(str(root)))
    indexed(home)
    sent = Filter("place", Operator.ANY, ("Ro\u0308sterei",))
    for asked in (written("place=R\u00f6sterei"), written("place=Ro\u0308sterei"), sent):
        result = search(home, SearchKeywords("beans", filters=(asked,)))
        assert [hit.id for hit in result.hits] == ["cafes/decomposed.md"]


def two_collections(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "papers" / "pigment.md", "# More pigment\n\nPigment and binder again.\n")
    add(home, AddCollection(str(tmp_path / "papers")))
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())


Timed = Callable[[Path, float | None, Callable[[], float]], SearchResult]


def meaning_timed(home: Path, budget: float | None, clock: Callable[[], float]) -> SearchResult:
    return vsearch(home, SearchVectors("pigment binder", budget=budget), Hashed(), watching=Watching(clock=clock))


def both_timed(home: Path, budget: float | None, clock: Callable[[], float]) -> SearchResult:
    return query(home, SearchHybrid("pigment binder", budget=budget), Hashed(), watching=Watching(clock=clock))


def like_timed(home: Path, budget: float | None, clock: Callable[[], float]) -> SearchResult:
    return similar(home, FindSimilar("notes/loose/plain.md", budget=budget), watching=Watching(clock=clock))


def later() -> Callable[[], float]:
    ticks = itertools.count(0.0, 10.0)
    return lambda: next(ticks)


@pytest.mark.parametrize("found", [meaning_timed, both_timed, like_timed])
def test_a_search_by_meaning_out_of_time_leaves_the_stores_it_has_not_read_out_and_says_so(
    home: Path, tmp_path: Path, found: Timed
) -> None:
    two_collections(home, tmp_path)
    whole = found(home, None, later())
    assert (whole.partial, whole.left_out, whole.searched) == (False, (), 2)
    cut = found(home, 1.0, later())
    assert (cut.partial, cut.searched) == (True, 1)
    assert [(each.collection, each.code) for each in cut.left_out] == [("papers", Problem.OUT_OF_TIME)]
    assert {hit.collection for hit in cut.hits} == {"notes"}


def test_passages_without_vectors_are_said_to_be_passed_over_by_meaning(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "notes" / "loose" / "new.md", "# New pigment\n\nPigment and binder, new.\n")
    indexed(home)
    for result in (vsearch(home, SearchVectors("pigment"), Hashed()), query(home, SearchHybrid("pigment"), Hashed())):
        (note,) = result.left_out
        assert (note.collection, note.code, note.message) == (
            "notes",
            Problem.NOT_EMBEDDED,
            "1 of its passages holds no vector yet: a search by meaning passes it over",
        )
        assert note.remedy == "embed the collection"
    assert search(home, SearchKeywords("pigment")).left_out == ()
