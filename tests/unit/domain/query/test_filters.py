import pytest

from sift.api.errors import InvalidRequestError
from sift.api.filters import Filter, FilterValue, Operator
from sift.domain.query.filters import Linked, LinkFilter, condition, linkings

ANY = Operator.ANY


def one(field: str, operator: Operator, *values: FilterValue) -> Filter:
    return Filter(field, operator, values)


def test_nothing_asked_restricts_nothing_and_a_path_alone_restricts_by_path() -> None:
    assert condition(None, [], reads_quotes=True) is None
    asked = condition("papers/*", [], reads_quotes=True)
    assert asked is not None
    assert (asked.path, asked.fields) == ("papers/*", "[]")


@pytest.mark.parametrize(
    "field, operator, values, refusal, part",
    [
        ("", ANY, (1,), "names no field", "field"),
        ("year", ANY, (), "'year' has no value", "values"),
        ("year", Operator.ABOVE, (1, 2), "'year' takes one value", "values"),
        ("year", ANY, (10**30,), "too large", "values"),
        ("year", Operator.ABOVE, (float("nan"),), "no finite number", "values"),
    ],
)
def test_a_filter_that_cannot_be_asked_is_refused_where_it_is_made(
    field: str, operator: Operator, values: tuple[FilterValue, ...], refusal: str, part: str
) -> None:
    with pytest.raises(InvalidRequestError, match=refusal) as refused:
        one(field, operator, *values)
    assert (refused.value.field, bool(refused.value.remedy)) == (f"filters.{part}", True)


def test_a_field_name_with_a_quote_is_refused_where_sqlite_reads_none() -> None:
    with pytest.raises(InvalidRequestError, match="double quote"):
        condition(None, [one('say "hi"', ANY, "yes")], reads_quotes=False)


def test_wikilink_values_of_any_and_none_are_the_links_to_follow() -> None:
    asked = [
        one("org", ANY, "[[Acme AG|Acme]]", "Beta", "[[a]] b]]"),
        one("org", Operator.NONE, " [[Beta#History]] "),
        one("org", ANY, "[[Acme AG]]"),
        one("year", Operator.ABOVE, "[[2024]]"),
    ]
    assert linkings(asked) == (LinkFilter("org", ANY, "Acme AG"), LinkFilter("org", Operator.NONE, "Beta"))


def test_a_search_kept_to_some_documents_asks_for_a_condition() -> None:
    asked = condition(None, [], reads_quotes=True, linked=Linked(within=[3, 1, 3]))
    assert asked is not None
    assert (asked.path, asked.within, asked.without) == (None, "[1, 3]", None)
    left = condition(None, [], reads_quotes=True, linked=Linked(without=[2]))
    assert left is not None
    assert (left.within, left.without) == (None, "[2]")


def test_a_path_written_with_backslashes_is_refused_with_how_to_write_it() -> None:
    with pytest.raises(InvalidRequestError, match="holds a backslash") as refused:
        condition("2024\\*", (), reads_quotes=True)
    remedy = "write the folders of a path in a collection with / between them"
    assert (refused.value.remedy, refused.value.field) == (remedy, "path")
