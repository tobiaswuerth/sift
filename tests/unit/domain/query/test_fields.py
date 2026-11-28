import pytest

from sift.api.errors import InvalidRequestError
from sift.api.fields import Fields
from sift.domain.query.fields import checked, order, picked

HELD: Fields = {"title": "Halo", "year": 2024, "tags": ["a", "b"]}


def test_the_fields_asked_for_are_picked_and_every_one_but_the_title_for_a_star() -> None:
    assert picked(HELD, ("year", "missing")) == {"year": 2024}
    assert picked(HELD, ("*",)) == {"year": 2024, "tags": ["a", "b"]}
    assert picked(HELD, ()) == {}


def test_an_order_by_a_field_puts_numbers_then_truths_then_text_and_those_without_it_last_as_they_came() -> None:
    each: list[Fields] = [{"r": "b"}, {}, {"r": 2}, {"r": None}, {"r": "a"}, {"r": 1.5}, {"r": True}]
    assert order(each, "r") == [5, 2, 6, 4, 0, 1, 3]
    assert order(each, "-r") == [0, 4, 6, 2, 5, 1, 3]


def test_sort_orders_values_as_range_filters_read_them() -> None:
    each: list[Fields] = [{"c": 2025}, {"c": "2024-06-15"}, {"c": "+4"}, {"c": 2023}, {"c": " 7 "}, {"c": "1.2.3"}]
    assert order(each, "c") == [2, 4, 3, 1, 0, 5]


def test_dates_written_without_leading_zeros_or_with_slashes_are_ordered_as_dates() -> None:
    each: list[Fields] = [
        {"d": "2024-1-15"},
        {"d": "2024-02-01"},
        {"d": "2024-12-01"},
        {"d": "2024/1/5"},
        {"d": "2024"},
        {"d": "2023-11-9"},
    ]
    assert order(each, "d") == [5, 4, 3, 0, 1, 2]


def test_a_list_is_ordered_by_its_first_value() -> None:
    each: list[Fields] = [{"t": ["b", "a"]}, {"t": [3, 9]}, {"t": []}, {"t": ["a"]}, {"t": {"k": 1}}]
    assert order(each, "t") == [1, 2, 3, 0, 4]


def test_a_sort_naming_no_field_is_refused() -> None:
    with pytest.raises(InvalidRequestError, match="names no field to sort by") as refused:
        checked("-")
    assert refused.value.remedy == "name a front matter field, e.g. created, or -created for the newest first"
    checked("-created")


def test_a_number_written_as_text_is_ordered_among_the_numbers_as_a_filter_reads_it() -> None:
    each: list[Fields] = [{"y": 2021}, {"y": "2019"}, {"y": 2024}, {"y": "2020.5"}, {"y": "2020s"}, {"y": "-3"}]
    assert order(each, "y") == [5, 1, 3, 0, 2, 4]


def test_whole_numbers_too_large_for_a_float_keep_their_order() -> None:
    each: list[Fields] = [{"r": 2**53 + 1}, {"r": 2**53}, {"r": 2.5}]
    assert order(each, "r") == [2, 1, 0]


def test_a_whole_number_of_more_digits_than_python_reads_is_ordered_as_larger_than_any_other() -> None:
    huge = "1" * 5_000
    each: list[Fields] = [{"n": huge}, {"n": "7"}, {"n": f"-{huge}"}, {"n": f"9{huge}-01-01"}, {"n": 10**300}]
    assert order(each, "n") == [2, 1, 4, 0, 3]
    assert order(each, "-n") == [3, 0, 4, 1, 2]
