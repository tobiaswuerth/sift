import pytest

from sift.api.errors import InvalidRequestError
from sift.api.filters import Filter, FilterValue, Operator, gathered, written
from sift.api.search import SearchKeywords
from sift.api.wire import decoded
from sift.domain.query.filters import LIST_REMEDY, condition, forms


@pytest.mark.parametrize(
    "text, expected",
    [
        ("year=2024", Filter("year", Operator.ANY, ("2024",))),
        ("year!=2023", Filter("year", Operator.NONE, ("2023",))),
        ("version=1.10", Filter("version", Operator.ANY, ("1.10",))),
        ("score=NaN", Filter("score", Operator.ANY, ("NaN",))),
        ("id=18446744073709551616", Filter("id", Operator.ANY, ("18446744073709551616",))),
        ("created>=2024-06", Filter("created", Operator.AT_LEAST, ("2024-06",))),
        ("created<2025", Filter("created", Operator.BELOW, (2025,))),
        ("score<=0.5", Filter("score", Operator.AT_MOST, (0.5,))),
        ("rank>3", Filter("rank", Operator.ABOVE, (3,))),
        ("open=true", Filter("open", Operator.ANY, (True,))),
        ('code="007"', Filter("code", Operator.ANY, ("007",))),
        ("journal-ref=Nature 12", Filter("journal-ref", Operator.ANY, ("Nature 12",))),
        ("date created >= 2024-05", Filter("date created", Operator.AT_LEAST, ("2024-05",))),
        ("note=a=b", Filter("note", Operator.ANY, ("a=b",))),
        ('tags=[1, "a", 2.50]', Filter("tags", Operator.ANY, ("1", "a", "2.50"))),
        ("org=[[Acme AG]]", Filter("org", Operator.ANY, ("[[Acme AG]]",))),
        ("n=" + "[" * 5000, Filter("n", Operator.ANY, ("[" * 5000,))),
    ],
)
def test_a_filter_is_written_as_a_field_an_operator_and_a_value_whose_numbers_stay_as_written_but_in_a_range(
    text: str, expected: Filter
) -> None:
    assert written(text) == expected


@pytest.mark.parametrize("text", ["year", "=2024", "year=", ">=5", "year=null", "year=[]", 'a={"b": 1}'])
def test_a_filter_written_otherwise_is_refused_saying_how_to_write_one(text: str) -> None:
    with pytest.raises(InvalidRequestError, match="KEY=VALUE") as refused:
        written(text)
    assert refused.value.remedy is not None


def test_values_of_one_field_asked_alike_are_gathered_into_any_or_none_of_them() -> None:
    found = gathered(
        tuple(written(text) for text in ("year=2023", "c!=a", "year=2024", "c!=b", "year>=2020", "year<2030"))
    )
    assert found == (
        Filter("year", Operator.ANY, ("2023", "2024")),
        Filter("c", Operator.NONE, ("a", "b")),
        Filter("year", Operator.AT_LEAST, (2020,)),
        Filter("year", Operator.BELOW, (2030,)),
    )


DECOMPOSED_CITY, COMPOSED_CITY = "Zu\u0308rich", "Z\u00fcrich"
DECOMPOSED_FIELD, COMPOSED_FIELD = "Ma\u0308rz", "M\u00e4rz"


def test_a_filter_built_from_json_composes_its_field_and_text_values() -> None:
    sent = {"field": DECOMPOSED_FIELD, "operator": "=", "values": [DECOMPOSED_CITY, 5]}
    asked = decoded(SearchKeywords, {"query": "x", "filters": [sent]})
    assert asked.filters == (Filter(COMPOSED_FIELD, Operator.ANY, (COMPOSED_CITY, 5)),)
    assert (asked.filters[0].field, asked.filters[0].values[0]) == (COMPOSED_FIELD, COMPOSED_CITY)


@pytest.mark.parametrize(
    "field, operator, value",
    [
        ("n", Operator.AT_LEAST, "5"),
        ("n", Operator.BELOW, "1e3"),
        ("city", Operator.ANY, DECOMPOSED_CITY),
        ("version", Operator.ANY, "1.10"),
        ("created", Operator.AT_LEAST, "2024-06"),
    ],
)
def test_a_filter_built_from_values_means_what_the_written_filter_means(
    field: str, operator: Operator, value: str
) -> None:
    built, typed = Filter(field, operator, (value,)), written(f"{field}{operator}{value}")
    assert condition(None, [built], reads_quotes=True) == condition(None, [typed], reads_quotes=True)
    assert built == typed


def test_a_range_of_a_number_too_large_or_not_finite_is_refused_or_kept_as_text() -> None:
    with pytest.raises(InvalidRequestError, match="is too large"):
        Filter("n", Operator.AT_LEAST, (str(2**63),))
    assert Filter("n", Operator.AT_LEAST, ("1e999",)).values == ("1e999",)
    assert Filter("n", Operator.AT_LEAST, ("NaN",)).values == ("NaN",)
    assert Filter("n", Operator.AT_LEAST, ("true",)).values == ("true",)


@pytest.mark.parametrize(
    "value, expected",
    [
        (str(2**63 - 1), [str(2**63 - 1), 2**63 - 1]),
        (str(-(2**63)), [str(-(2**63)), -(2**63)]),
        (str(2**63), [str(2**63), float(2**63)]),
        (str(-(2**63) - 1), [str(-(2**63) - 1), float(-(2**63) - 1)]),
        (2**63 - 1, [2**63 - 1, str(2**63 - 1)]),
        ("1e3", ["1e3", 1000.0]),
        ("1.10", ["1.10", 1.1]),
        ("007", ["007"]),
        ("1e999", ["1e999"]),
        ("NaN", ["NaN"]),
        ("True", ["true", "True", "TRUE"]),
        (False, ["false", "False", "FALSE"]),
    ],
)
def test_a_value_matches_as_written_and_as_the_number_or_truth_front_matter_reads_it_as(
    value: FilterValue, expected: list[object]
) -> None:
    assert forms(value) == expected


def test_a_value_front_matter_reads_as_a_list_is_refused_saying_how_to_give_several() -> None:
    with pytest.raises(InvalidRequestError, match="reads as a list") as refused:
        forms("[a, c]")
    assert refused.value.remedy == LIST_REMEDY
    assert forms("[[Acme AG]]") == ["[[Acme AG]]"]


@pytest.mark.parametrize(
    "field, values, said",
    [
        ("f" * 1_001, ("x",), "field is 1,001 characters long, more than 1,000"),
        ("f", ("x",) * 100_001, "has 100,001 values, more than 100,000"),
        ("f", ("x" * 10_001,), "has a value 10,001 characters long, more than 10,000"),
    ],
)
def test_a_filter_too_long_is_refused_saying_how_long_it_may_be(field: str, values: tuple[str, ...], said: str) -> None:
    with pytest.raises(InvalidRequestError, match=said):
        Filter(field, Operator.ANY, values)
    assert Filter("f" * 1_000, Operator.ANY, ("x" * 10_000,) * 2).values[0] == "x" * 10_000
