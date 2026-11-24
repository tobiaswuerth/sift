import pytest

from sift.api.errors import InvalidRequestError
from sift.api.filters import Filter, Operator, gathered, written


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
