import pytest

from sift.api.errors import InvalidRequestError
from sift.api.filters import Filter, FilterValue, Operator
from sift.domain.query.filters import Linking, condition, linkings

ANY = Operator.ANY


def one(field: str, operator: Operator, *values: FilterValue) -> Filter:
    return Filter(field, operator, values)


def test_nothing_asked_restricts_nothing_and_a_path_alone_restricts_by_path() -> None:
    assert condition(None, [], reads_quotes=True) is None
    asked = condition("papers/*", [], reads_quotes=True)
    assert asked is not None
    assert (asked.path, asked.fields) == ("papers/*", "[]")


@pytest.mark.parametrize(
    "filters, refusal",
    [
        ([one("", ANY, 1)], "names no field"),
        ([one("year", ANY)], "'year' has no value"),
        ([one("year", Operator.ABOVE, 1, 2)], "'year' takes one value"),
        ([one("year", ANY, 10**30)], "too large"),
        ([one("year", Operator.ABOVE, float("nan"))], "no finite number"),
    ],
)
def test_a_filter_that_cannot_be_asked_is_refused_saying_how_to_filter(filters: list[Filter], refusal: str) -> None:
    with pytest.raises(InvalidRequestError, match=refusal) as refused:
        condition(None, filters, reads_quotes=True)
    assert refused.value.remedy is not None


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
    assert linkings(asked) == (Linking("org", ANY, "Acme AG"), Linking("org", Operator.NONE, "Beta"))


def test_a_search_kept_to_some_documents_asks_for_a_condition() -> None:
    asked = condition(None, [], reads_quotes=True, within=[3, 1, 3])
    assert asked is not None
    assert (asked.path, asked.within) == (None, "[1, 3]")
