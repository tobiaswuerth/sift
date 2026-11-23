from collections.abc import Callable
from pathlib import Path

import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError
from sift.api.filters import Filter, Operator
from sift.api.indexing import IndexCollections
from sift.api.search import SearchHybrid, SearchKeywords, SearchResult, SearchVectors
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.app.query import query
from sift.app.search import search
from sift.app.vsearch import vsearch
from tests.models import Hashed

DAYS = ("2023-11-30", "2024-02-10", "2024-06-15", "2024-12-31", "2025-03-01")
ANY, NONE = Operator.ANY, Operator.NONE


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def home(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    for n, day in enumerate(DAYS):
        fields = f"created: {day}\nrank: {n}\ntags: [paint, {'oil' if n % 2 else 'water'}]\n"
        write(root / "dated" / f"{day}.md", f"---\n{fields}---\n# Pigment {day}\n\nPigment and binder, part {n}.\n")
    write(root / "loose" / "pigment.md", "---\nrank: high\n---\n# Pigment notes\n\nPigment pigment pigment.\n")
    write(root / "loose" / "plain.md", "# Plain\n\nPigment without front matter.\n")
    home = tmp_path / "home"
    add(home, AddCollection(str(root)))
    index(home, IndexCollections(), lambda _: None)
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    return home


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


def test_a_filter_leaves_as_many_results_as_asked_for_when_better_matches_fail_it(home: Path) -> None:
    oil = (Filter("tags", ANY, ("oil",)),)
    assert len(search(home, SearchKeywords("pigment", k=2, filters=oil)).hits) == 2
    assert len(vsearch(home, SearchVectors("pigment", k=2, filters=oil), Hashed()).hits) == 2


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
        (SearchKeywords("pigment", filters=(Filter("year", Operator.ABOVE, (1, 2)),)), "takes one value"),
        (SearchKeywords("pigment", sort="-"), "sort names no field"),
        (SearchKeywords("pigment", fields=("",)), "names no field"),
    ],
)
def test_options_that_cannot_be_followed_are_refused_saying_how(
    home: Path, request_: SearchKeywords, refusal: str
) -> None:
    with pytest.raises(InvalidRequestError, match=refusal) as refused:
        search(home, request_)
    assert refused.value.remedy is not None
