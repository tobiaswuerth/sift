import json
from pathlib import Path
from typing import cast

import pytest

from sift.api.collections import AddCollection, ShowStatus, TitleFrom
from sift.api.documents import GetDocuments, WantedDocument
from sift.api.errors import (
    InternalError,
    InvalidRequestError,
    ModelUnavailableError,
    NotFoundError,
    Problem,
    SiftError,
    StoreUnavailableError,
)
from sift.api.filters import Filter, Operator
from sift.api.indexing import CollectionIndexed, Indexed
from sift.api.links import Direction, FindMentions, FindSimilar, ListUnresolved, ShowLinks
from sift.api.notices import Notice
from sift.api.search import Hit, SearchHybrid, SearchKeywords, SearchResult, SearchVectors
from sift.api.wire import Json, decoded, decoded_error, encoded, encoded_error, read_answer
from tests.answers import SAMPLES

REQUESTS = (
    AddCollection(
        str(Path("/data/notes").absolute()),
        name="n",
        title_from=TitleFrom.NAME,
        nested=True,
        text_fields=("body", "abstract"),
    ),
    ShowStatus(("notes",), unreadable=True),
    GetDocuments((WantedDocument("notes/a.md", 2, 4), WantedDocument("b.md")), fields=("*",)),
    SearchKeywords(
        "halo",
        k=3,
        collections=("notes",),
        filters=(Filter("year", Operator.AT_LEAST, (2020,)), Filter("tags", Operator.ANY, ("a", 1, 2.5, True))),
        sort="-year",
        budget=0.5,
    ),
    SearchVectors("halo", offset=10, compact=True),
    SearchHybrid("halo", candidates=7, links_to=("Acme",)),
    ShowLinks("Acme", direction=Direction.BOTH, depth=2, field="org"),
    ListUnresolved(("notes",), k=5),
    FindMentions("notes/Acme.md", also=("emails",)),
    FindSimilar("notes/a.md", all_passages=True),
)


@pytest.mark.parametrize("request_", REQUESTS, ids=[type(each).__name__ for each in REQUESTS])
def test_every_request_comes_back_from_its_json_as_it_was(request_: object) -> None:
    assert decoded(type(request_), cast("object", json.loads(json.dumps(encoded(request_))))) == request_


@pytest.mark.parametrize("name", list(SAMPLES))
def test_every_answer_comes_back_from_its_json_as_it_was(name: str) -> None:
    answer = SAMPLES[name]
    assert decoded(type(answer), cast("object", json.loads(json.dumps(encoded(answer))))) == answer


def test_what_a_search_left_out_comes_back_too() -> None:
    found = SearchResult(
        (), 1, (), left_out=(Notice(Problem.OUT_OF_TIME, "out of time", "give more", "notes"),), scored_by="keywords"
    )
    assert decoded(SearchResult, encoded(found)) == found


ANY_VALUE = "text, a whole number, a finite number or true or false"
REFUSED_JSON = {
    "a value of another kind": (
        {"k": "10"},
        ("k", "k is '10', not a whole number", "give k as a whole number"),
    ),
    "text for a list": (
        {"collections": "notes"},
        ("collections", "collections is 'notes', not a list of text", "give collections as a list of text"),
    ),
    "a field the request lacks": ({"kk": 3}, ("kk", "the request takes no 'kk'", "leave out 'kk'")),
    "a field left out": ({"query": None}, ("query", "query is None, not text", "give query as text")),
    "an operator of none": (
        {"filters": [{"field": "y", "operator": "~", "values": [1]}]},
        ("filters[0].operator", "'~' is none of =, !=, <, <=, >, >=", "give filters[0].operator as one of"),
    ),
    "a filter without its operator": (
        {"filters": [{"field": "y", "values": [1]}]},
        ("filters[0].operator", "filters[0] needs operator", "give filters[0].operator"),
    ),
    "a filter written as text": (
        {"filters": ["y=1"]},
        ("filters[0]", "filters[0] is 'y=1', not a filter", "give filters[0] as a filter"),
    ),
    "a filter value of no kind": (
        {"filters": [{"field": "y", "operator": "=", "values": [1, [2]]}]},
        ("filters[0].values[1]", f"filters[0].values[1] is [2], not {ANY_VALUE}", "give filters[0].values[1] as"),
    ),
    "values not in a list": (
        {"filters": [{"field": "y", "operator": "=", "values": "x"}]},
        (
            "filters[0].values",
            f"filters[0].values is 'x', not a list of values, each {ANY_VALUE}",
            "give filters[0].values as a list of values",
        ),
    ),
    "a range of two values": (
        {
            "filters": [
                {"field": "y", "operator": "=", "values": [1]},
                {"field": "y", "operator": "<", "values": [1, 2]},
            ]
        },
        ("filters[1].values", "the range on 'y' takes one value, not 2", "give a field, an operator"),
    ),
    "a document's line of another kind": (
        {"documents": [{"id": "a"}, {"id": "b", "first_line": "1"}]},
        ("documents[1].first_line", "documents[1].first_line is '1', not a whole number or nothing", "give"),
    ),
    "a document's line out of bounds": (
        {"documents": [{"id": "a", "last_line": 0}]},
        ("documents[0].last_line", "0 is outside 1 to 2,147,483,647", "give 1 to 2,147,483,647"),
    ),
}


@pytest.mark.parametrize("given, refusal", REFUSED_JSON.values(), ids=REFUSED_JSON.keys())
def test_json_that_makes_no_request_is_refused_naming_the_path_at_fault_and_what_to_give(
    given: dict[str, object], refusal: tuple[str, str, str]
) -> None:
    kind = GetDocuments if "documents" in given else SearchKeywords
    with pytest.raises(InvalidRequestError) as refused:
        decoded(kind, {"query": "x", **given} if kind is SearchKeywords else given)
    field, said, remedy = refusal
    assert (refused.value.field, refused.value.message, refused.value.remedy.startswith(remedy)) == (field, said, True)


def test_a_request_without_a_field_it_needs_is_refused_naming_it() -> None:
    with pytest.raises(InvalidRequestError) as refused:
        decoded(SearchKeywords, {})
    assert (refused.value.field, refused.value.message, refused.value.remedy) == (
        "query",
        "the request needs query",
        "give query",
    )


@pytest.mark.parametrize("value", ["x" * 500_000, 10**400, ["x" * 500_000]], ids=["text", "number", "list"])
def test_a_value_refused_is_echoed_cut_short(value: object) -> None:
    with pytest.raises(InvalidRequestError) as refused:
        decoded(SearchKeywords, {"query": "x", "k": value})
    assert len(refused.value.message) < 200
    assert "..." in refused.value.message


def test_what_has_no_json_is_a_bug_said_at_once() -> None:
    with pytest.raises(TypeError, match="has no JSON"):
        encoded(object())
    with pytest.raises(TypeError, match="is no error"):
        decoded_error(["busy"])


def test_an_answer_of_another_state_is_refused() -> None:
    with pytest.raises(InvalidRequestError, match="state is 'skipped', not indexed") as refused:
        decoded(Indexed, {"files": 1, "added": 1, "changed": 0, "removed": 0, "unreadable": 0, "state": "skipped"})
    assert refused.value.field == "state"


@pytest.mark.parametrize(
    "outcome",
    [3, "indexed", {"state": "gone", "message": "m", "remedy": None}],
)
def test_a_value_none_of_its_union_s_kinds_takes_is_refused(outcome: object) -> None:
    with pytest.raises(InvalidRequestError, match=r"outcome is .*, not an indexed or a skipped") as refused:
        decoded(CollectionIndexed, {"name": "n", "outcome": outcome})
    assert refused.value.field == "outcome"


def test_a_request_is_read_from_an_object_alone() -> None:
    with pytest.raises(InvalidRequestError, match="a request is an object"):
        decoded(SearchKeywords, ["halo"])


@pytest.mark.parametrize(
    "error",
    [
        InvalidRequestError("k is 0", "give 1 to 10", "k"),
        NotFoundError("no collection 'x'", "add it", code=Problem.UNKNOWN_COLLECTION),
        StoreUnavailableError("damaged", "remove it", code=Problem.STORE_DAMAGED),
        ModelUnavailableError("gone", "check the name"),
    ],
)
def test_an_error_comes_back_from_its_json_as_it_was(error: SiftError) -> None:
    back = decoded_error(cast("Json", json.loads(json.dumps(encoded_error(error)))))
    assert (type(back), back.message, back.remedy, back.code) == (type(error), error.message, error.remedy, error.code)
    if isinstance(error, InvalidRequestError):
        assert isinstance(back, InvalidRequestError)
        assert back.field == error.field


@pytest.mark.parametrize(
    "score, shown", [(0.7465865612030029, 0.7465866), (12.345678901, 12.34568), (1e-12, 1e-12), (0.0, 0.0)]
)
def test_a_hit_s_score_carries_the_digits_a_float32_holds_and_no_more(score: float, shown: float) -> None:
    hit = Hit("n/a.md", "n", "a.md", "A", 1, 2, score, None)
    assert (hit.score, json.dumps(cast("dict[str, Json]", encoded(hit))["score"])) == (shown, json.dumps(shown))


def test_an_answer_of_another_shape_is_refused_naming_the_kind_it_wanted() -> None:
    hit = Hit("n/a.md", "n", "a.md", "A", 1, 2, 0.5, None)
    sent = cast("dict[str, Json]", encoded(SearchResult((hit,), 1, (), scored_by="keywords")))
    cast("list[dict[str, Json]]", sent["hits"])[0]["fields"] = 3
    with pytest.raises(InvalidRequestError) as refused:
        read_answer(SearchResult, sent)
    assert (refused.value.field, refused.value.message) == (
        "hits[0].fields",
        "hits[0].fields is 3, not an object or nothing",
    )


def test_an_answer_of_a_newer_sift_is_read_with_what_it_adds_left_out_but_a_request_is_not() -> None:
    hit = Hit("n/a.md", "n", "a.md", "A", 1, 2, 0.5, None)
    sent = cast("dict[str, Json]", encoded(SearchResult((hit,), 1, (), scored_by="keywords")))
    sent["total"] = 9
    cast("list[dict[str, Json]]", sent["hits"])[0]["kind"] = "passage"
    assert read_answer(SearchResult, sent) == SearchResult((hit,), 1, (), scored_by="keywords")
    with pytest.raises(InvalidRequestError, match="the request takes no 'total'"):
        decoded(SearchKeywords, {"query": "x", "total": 9})


@pytest.mark.parametrize(
    "sent, kind, code, remedy",
    [
        (
            {"kind": "store_unavailable", "code": "new_code", "message": "m", "remedy": "r", "field": None},
            StoreUnavailableError,
            Problem.STORE_UNAVAILABLE,
            "r",
        ),
        ({"kind": "new_kind", "code": "new_code", "message": "m", "remedy": "r"}, InternalError, Problem.INTERNAL, "r"),
        ({"kind": "not_found", "code": "unknown_note", "message": "m"}, NotFoundError, Problem.UNKNOWN_NOTE, ""),
    ],
)
def test_an_error_of_a_newer_sift_is_read_as_its_kind_or_as_internal(
    sent: dict[str, Json], kind: type[SiftError], code: Problem, remedy: str
) -> None:
    found = decoded_error(sent)
    assert (type(found), found.message, found.remedy, found.code) == (kind, "m", remedy, code)
