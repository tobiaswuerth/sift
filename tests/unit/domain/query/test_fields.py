import pytest

from sift.api.errors import InvalidRequestError
from sift.api.fields import Fields
from sift.domain.query.fields import checked, order, picked

HELD: Fields = {"title": "Halo", "year": 2024, "tags": ["a", "b"]}


def test_the_fields_asked_for_are_picked_and_every_one_but_the_title_for_a_star() -> None:
    assert picked(HELD, ("year", "missing")) == {"year": 2024}
    assert picked(HELD, ("*",)) == {"year": 2024, "tags": ["a", "b"]}
    assert picked(HELD, ()) == {}


def test_an_order_by_a_field_puts_numbers_before_text_and_those_without_it_last_as_they_came() -> None:
    each: list[Fields] = [{"r": "b"}, {}, {"r": 2}, {"r": None}, {"r": "a"}, {"r": 1.5}, {"r": True}]
    assert order(each, "r") == [6, 5, 2, 4, 0, 1, 3]
    assert order(each, "-r") == [0, 4, 2, 5, 6, 1, 3]


@pytest.mark.parametrize(
    "fields, sort, refusal",
    [
        (("url", ""), None, "names no field between two commas"),
        ((), "-", "the sort names no field"),
        ((), "", "the sort"),
    ],
)
def test_fields_or_a_sort_naming_no_field_are_refused(fields: tuple[str, ...], sort: str | None, refusal: str) -> None:
    with pytest.raises(InvalidRequestError, match=refusal) as refused:
        checked(fields, sort)
    assert refused.value.remedy is not None
    checked(("url",), "-created")
