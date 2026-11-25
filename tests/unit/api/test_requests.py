import dataclasses
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast, get_args, get_origin, get_type_hints

import pytest

from sift.api import collections, documents, embedding, indexing, links, search
from sift.api.described import Bounds, Help
from sift.api.errors import InvalidRequestError
from sift.api.filters import Filter, Operator
from sift.api.requests import Request
from sift.api.search import SearchHybrid, SearchKeywords
from sift.api.wire import decoded

ABSOLUTE = str(Path.cwd())


def built(kind: Callable[..., object], *given: object, **named: object) -> object:
    return kind(*given, **named)


@dataclass(frozen=True, slots=True)
class Asked(Request):
    k: Annotated[int, Help("how many"), Bounds(1, 9)] = 5
    name: Annotated[str, Help("a name")] = "x"
    share: Annotated[float | None, Help("a share")] = None
    plain: int = 0


@pytest.mark.parametrize("k, said", [(0, "0 is outside 1 to 9"), (10, "10 is outside 1 to 9")])
def test_a_field_outside_its_bounds_is_refused_by_name(k: int, said: str) -> None:
    with pytest.raises(InvalidRequestError, match=said) as refused:
        Asked(k=k)
    assert (refused.value.field, refused.value.remedy) == ("k", "give 1 to 9")


def test_fields_within_their_bounds_and_of_their_kinds_pass() -> None:
    assert (Asked(k=1).k, Asked(k=9).k, Asked(share=1).share, Asked(share=0.5).share) == (1, 9, 1, 0.5)


@pytest.mark.parametrize(
    "asked, field, said",
    [
        ({"k": "10"}, "k", "k is '10', not a whole number"),
        ({"k": True}, "k", "k is True, not a whole number"),
        ({"k": 2.5}, "k", "k is 2.5, not a whole number"),
        ({"name": None}, "name", "name is None, not text"),
        ({"share": "1"}, "share", "share is '1', not a finite number or nothing"),
        ({"share": False}, "share", "share is False, not a finite number or nothing"),
    ],
)
def test_a_field_of_another_kind_is_refused_by_name_saying_what_it_takes(
    asked: dict[str, object], field: str, said: str
) -> None:
    with pytest.raises(InvalidRequestError, match=said) as refused:
        built(Asked, **asked)
    assert refused.value.field == field


@pytest.mark.parametrize(
    "asked, field, said",
    [
        ({"collections": "notes"}, "collections", "collections is 'notes', not a list of text"),
        ({"collections": ["notes"]}, "collections", r"collections is \['notes'\], not a list of text"),
        ({"fields": ("url", 3)}, "fields", r"fields is \('url', 3\), not a list of text"),
        ({"filters": ({"year": 2024},)}, "filters", "not a list of filters"),
        ({"budget": "1"}, "budget", "budget is '1', not a finite number or nothing"),
        ({"budget": float("nan")}, "budget", "budget is nan, not a finite number or nothing"),
        ({"budget": float("inf")}, "budget", "budget is inf, not a finite number or nothing"),
        ({"all_passages": 1}, "all_passages", "all_passages is 1, not true or false"),
    ],
)
def test_a_search_of_fields_of_another_kind_is_refused_whichever_surface_built_it(
    asked: dict[str, object], field: str, said: str
) -> None:
    with pytest.raises(InvalidRequestError, match=said) as refused:
        built(SearchKeywords, "x", **asked)
    assert refused.value.field == field


def test_a_field_without_a_description_is_checked_too() -> None:
    with pytest.raises(InvalidRequestError, match="plain is 'x', not a whole number"):
        built(Asked, plain="x")


@dataclass(frozen=True, slots=True)
class Listed(Request):
    names: list[str]


def test_a_request_of_a_kind_no_surface_can_send_is_a_bug_said_at_once() -> None:
    with pytest.raises(TypeError, match=r"a request holds no list\[str\]"):
        Listed(["a"])


def test_a_choice_is_one_of_its_members() -> None:
    with pytest.raises(InvalidRequestError, match="title_from is 'name', not heading, name or nothing") as refused:
        built(collections.AddCollection, ABSOLUTE, title_from="name")
    assert refused.value.field == "title_from"
    assert collections.AddCollection(ABSOLUTE, title_from=collections.TitleFrom.NAME).title_from == "name"


@pytest.mark.parametrize(
    "operator, values, said",
    [
        ("=", ("Alpha",), "the operator '=' is no Operator"),
        (Operator.ANY, (None,), "None is no text, number, true or false"),
        (Operator.ANY, ([1],), r"\[1\] is no text, number, true or false"),
    ],
)
def test_a_filter_of_an_operator_or_value_of_another_kind_is_refused(
    operator: object, values: tuple[object, ...], said: str
) -> None:
    with pytest.raises(InvalidRequestError, match=said) as refused:
        built(Filter, "title", operator, values)
    assert refused.value.field == "filters"


def test_a_page_of_results_lies_within_what_a_search_may_find() -> None:
    assert SearchHybrid("x", k=10, offset=9_990).offset == 9_990
    with pytest.raises(InvalidRequestError, match="-1 is outside 0 to 9,999") as refused:
        SearchHybrid("x", offset=-1)
    assert refused.value.field == "offset"


def _described(kind: type) -> bool:
    hints = cast("dict[str, object]", get_type_hints(kind, include_extras=True))
    return any(
        get_origin(hint) is Annotated
        and any(isinstance(part, Help) for part in cast("tuple[object, ...]", get_args(hint))[1:])
        for hint in hints.values()
    )


def test_every_dataclass_of_sift_api_whose_fields_are_described_is_a_request_that_checks_them() -> None:
    modules = (collections, documents, embedding, indexing, links, search)
    described = [
        kind
        for module in modules
        for _, kind in inspect.getmembers(module, inspect.isclass)
        if kind.__module__ == module.__name__ and dataclasses.is_dataclass(kind) and _described(kind)
    ]
    assert len(described) >= 12
    assert [kind.__name__ for kind in described if not issubclass(kind, Request)] == []


HALF = "a" + chr(0xD800)


@pytest.mark.parametrize(
    "build, field",
    [
        (lambda: SearchKeywords(HALF), "query"),
        (lambda: SearchKeywords("x", path=HALF), "path"),
        (lambda: SearchKeywords("x", collections=("notes", HALF)), "collections"),
        (lambda: documents.WantedDocument(HALF), "id"),
        (lambda: links.ShowLinks(HALF), "id"),
        (lambda: Filter(HALF, Operator.ANY, ("x",)), "filters"),
        (lambda: Filter("year", Operator.ANY, (HALF,)), "filters"),
    ],
)
def test_text_with_half_a_surrogate_pair_is_refused_by_field(build: Callable[[], object], field: str) -> None:
    with pytest.raises(InvalidRequestError, match="half a surrogate pair") as refused:
        build()
    assert (refused.value.field, refused.value.remedy) == (field, "send it as text in UTF-8")


DECOMPOSED, COMPOSED = "Ro\u0308sterei", "R\u00f6sterei"


def test_the_names_of_fields_to_show_sort_by_or_look_in_are_composed_however_they_were_sent() -> None:
    asked = decoded(SearchKeywords, {"query": "x", "fields": [DECOMPOSED], "sort": DECOMPOSED})
    shown = documents.GetDocuments((documents.WantedDocument("notes/a.md"),), fields=(DECOMPOSED,))
    looked = links.FindMentions("notes/a.md", also=(DECOMPOSED, "plain"))
    found = (asked.fields, asked.sort, shown.fields, looked.also)
    assert found == ((COMPOSED,), COMPOSED, (COMPOSED,), (COMPOSED, "plain"))
    assert (SearchKeywords("x").sort, SearchKeywords(DECOMPOSED).query) == (None, DECOMPOSED)
