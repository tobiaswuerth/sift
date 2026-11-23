import pytest

from sift.api.errors import InvalidRequestError
from sift.api.filters import Filter, FilterValue, Operator
from sift.domain.query.filters import condition

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
