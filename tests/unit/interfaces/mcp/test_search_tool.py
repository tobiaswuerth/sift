from pathlib import Path
from typing import cast

import pytest

from sift.api.errors import InvalidRequestError
from sift.interfaces.mcp.answers import MOST_TEXT
from sift.interfaces.mcp.searches import MOST_QUERIES
from tests.agent_tools import Direct, Object, called, ids
from tests.files import write
from tests.homes import indexed


def test_a_search_is_hybrid_unless_another_mode_is_asked_for(direct: Direct) -> None:
    for mode, act in (({}, "query"), ({"mode": "keyword"}, "search"), ({"mode": "semantic"}, "vsearch")):
        assert ids(called(direct, "search", {"query": "Kestrel", "k": 1, **mode})) == ["open/Plan.md"]
        assert direct.asked[-1] == act


def test_a_hit_says_where_it_is_and_what_it_holds(direct: Direct) -> None:
    (hit,) = cast("list[Object]", called(direct, "search", {"query": "Kestrel", "mode": "keyword"})["hits"])
    assert set(hit) == {"id", "title", "first_line", "last_line", "score", "text"}
    assert (hit["id"], hit["title"], hit["first_line"], hit["last_line"]) == ("open/Plan.md", "Plan", 4, 6)
    compact = called(direct, "search", {"query": "Kestrel", "mode": "keyword", "compact": True, "fields": ["*"]})
    (shown,) = cast("list[Object]", compact["hits"])
    assert ("text" in shown, shown["fields"]) == (False, {"created": "2026-01-02"})
    bare = called(direct, "search", {"query": "osprey", "mode": "keyword", "compact": True, "fields": ["*"]})
    assert set(cast("list[Object]", bare["hits"])[0]) == {"id", "title", "first_line", "last_line", "score"}


def test_several_phrasings_are_searched_in_one_call(direct: Direct) -> None:
    answer = called(direct, "search", {"query": ["Kestrel", "osprey"], "mode": "keyword"})
    results = cast("list[Object]", answer["results"])
    assert [(each["query"], ids(each)) for each in results] == [
        ("Kestrel", ["open/Plan.md"]),
        ("osprey", ["open/Other.md"]),
    ]


@pytest.mark.parametrize(
    "arguments,wrong",
    [
        ({"query": "x", "mode": "fuzzy"}, "mode"),
        ({"query": []}, "query"),
        ({"query": ["x"] * (MOST_QUERIES + 1)}, "query"),
        ({"query": 3}, "query"),
        ({}, "query"),
        ({"query": "x", "mode": "keyword", "candidates": 5}, "candidates"),
        ({"query": "x", "k": 0}, "k"),
    ],
)
def test_a_wrong_search_is_refused_naming_its_field(direct: Direct, arguments: Object, wrong: str) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        called(direct, "search", arguments)
    assert caught.value.field == wrong


def test_an_answer_says_what_its_scores_are_by(direct: Direct) -> None:
    for mode, by in (("keyword", "keywords"), ("semantic", "vectors"), ("hybrid", "reranker")):
        assert called(direct, "search", {"query": "Kestrel", "mode": mode})["scored_by"] == by
    assert called(direct, "related", {"id": "open/Plan.md"})["scored_by"] == "vectors"


def test_the_texts_of_one_answer_s_hits_stay_within_what_an_answer_holds_whatever_is_asked(
    direct: Direct, tmp_path: Path
) -> None:
    for n in range(40):
        write(tmp_path / "open" / f"falcon{n:02}.md", f"# Falcon {n}\n\nfalcon " + "word " * 560 + "\n")
    indexed(direct.acting.settings.home)
    answer = called(direct, "search", {"query": ["falcon", "falcon word"], "mode": "keyword", "k": 30})
    hits = [hit for each in cast("list[Object]", answer["results"]) for hit in cast("list[Object]", each["hits"])]
    texts = [cast("str", hit["text"]) for hit in hits if "text" in hit]
    assert (len(hits), sum(map(len, texts)) <= MOST_TEXT, 30 < len(texts) < 40) == (60, True, True)
    first, second = cast("list[Object]", answer["results"])
    assert "notes" not in first
    full = f"this answer's text is at its {MOST_TEXT:,} characters"
    assert second["notes"] == [f"the text of {60 - len(texts)} hits is left out, as {full}: get them by id and lines"]


@pytest.mark.parametrize("queries, at", [(["Kestrel", ""], 1), (["", "Kestrel"], 0), (["Kestrel", "x" * 100_001], 1)])
def test_no_query_of_several_is_searched_when_one_of_them_is_wrong(direct: Direct, queries: list[str], at: int) -> None:
    with pytest.raises(InvalidRequestError) as refused:
        called(direct, "search", {"query": queries, "mode": "keyword"})
    assert (refused.value.field, refused.value.message.startswith(f"query[{at}] "), direct.asked) == (
        f"query[{at}]",
        True,
        [],
    )


def test_a_query_of_several_that_cannot_be_read_is_named_by_its_place(direct: Direct) -> None:
    with pytest.raises(InvalidRequestError, match="NOT needs words before and after it") as refused:
        called(direct, "search", {"query": ["Kestrel", "NOT a"], "mode": "keyword"})
    assert refused.value.field == "query[1]"
    with pytest.raises(InvalidRequestError) as alone:
        called(direct, "search", {"query": "NOT a", "mode": "keyword"})
    assert alone.value.field == "query"


def test_a_search_given_what_it_does_not_take_is_told_the_name_meant_or_what_it_takes(direct: Direct) -> None:
    with pytest.raises(InvalidRequestError) as near:
        called(direct, "search", {"query": "x", "collection": "open"})
    assert (near.value.field, near.value.message, near.value.remedy) == (
        "collection",
        "search takes no 'collection'",
        "give collections instead",
    )
    with pytest.raises(InvalidRequestError) as far:
        called(direct, "search", {"query": "x", "mode": "keyword", "zz": 1})
    assert far.value.remedy == (
        "leave out 'zz': it takes query, mode, k, collections, path, filters, fields, sort, links_to, linked_from, "
        "all_passages, compact, adaptive, offset, budget"
    )


def test_a_search_says_when_more_results_wait(direct: Direct) -> None:
    answer = called(direct, "search", {"query": "Note", "mode": "keyword", "k": 1})
    assert (len(ids(answer)), answer["more"]) == (1, True)
    assert "more" not in called(direct, "search", {"query": "Note", "mode": "keyword", "k": 5})


def test_several_queries_share_the_time_budget_one_call_has(direct: Direct) -> None:
    called(direct, "search", {"query": ["Kestrel", "osprey"], "mode": "keyword", "budget": 2.0})
    assert [cast("Object", each)["budget"] for each in direct.sent] == [1.0, 1.0]
    called(direct, "search", {"query": "Kestrel", "mode": "keyword", "budget": 2.0})
    assert cast("Object", direct.sent[-1])["budget"] == 2.0


@pytest.mark.parametrize(
    "arguments, said",
    [
        ({"query": ["a", "b"], "budget": -2}, "^-2 is outside"),
        ({"query": ["a"] * 20, "budget": 0.01}, r"0\.01 s shared by 20 queries"),
        ({"query": ["a", "b"], "k": 5_001}, r"10,000 hits"),
    ],
)
def test_a_budget_or_k_too_wrong_for_several_queries_is_refused_as_asked(
    direct: Direct, arguments: Object, said: str
) -> None:
    with pytest.raises(InvalidRequestError, match=said):
        called(direct, "search", arguments)
    assert direct.sent == []


def test_several_queries_may_ask_for_as_many_hits_in_all_as_one_search(direct: Direct) -> None:
    answer = called(direct, "search", {"query": ["Kestrel", "osprey"], "mode": "keyword", "k": 5_000})
    assert len(cast("list[Object]", answer["results"])) == 2
