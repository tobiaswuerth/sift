import pytest

from sift.api.fields import FieldValue
from sift.domain.text.json_values import (
    MOST_DEPTH,
    NestedTooDeeplyError,
    depth,
    inside,
    json_value,
    without_lone_surrogates,
)


@pytest.mark.parametrize(
    "written, read",
    [
        ("1.5", 1.5),
        ("1e400", "1e400"),
        ("NaN", "NaN"),
        ("[Infinity, -Infinity]", ["Infinity", "-Infinity"]),
        ('{"a": [1, "b", null, true]}', {"a": [1, "b", None, True]}),
        ('"\\ud83d\\ude00"', "\U0001f600"),
        ('{"k\\ud800": "\\udc00"}', {"k\N{REPLACEMENT CHARACTER}": "\N{REPLACEMENT CHARACTER}"}),
    ],
)
def test_json_values_are_kept_as_sqlite_can_store_them(written: str, read: FieldValue) -> None:
    assert json_value(written) == read


def test_json_nested_deeper_than_sqlite_stores_is_refused() -> None:
    assert depth(json_value("[" * MOST_DEPTH + "]" * MOST_DEPTH)) == MOST_DEPTH
    with pytest.raises(NestedTooDeeplyError, match=f"more than {MOST_DEPTH}"):
        json_value("[" * (MOST_DEPTH + 1) + "]" * (MOST_DEPTH + 1))


def test_json_too_deep_is_refused_as_a_value_error_before_python_parses_it() -> None:
    assert issubclass(NestedTooDeeplyError, ValueError)
    with pytest.raises(NestedTooDeeplyError, match=f"more than {MOST_DEPTH}"):
        json_value("[" * 100_000)
    brackets = "[" * 2000
    assert json_value(f'["\\"{brackets}"]') == [f'"{brackets}']


def test_a_lone_surrogate_is_replaced_at_the_deepest_level_sqlite_stores() -> None:
    nesting = MOST_DEPTH - 1
    found = json_value("[" * nesting + '"\\ud800"' + "]" * nesting)
    for _ in range(nesting):
        held = inside(found)
        assert held is not None
        (found,) = held
    assert found == "\N{REPLACEMENT CHARACTER}"


def test_what_is_no_json_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="Expecting value"):
        json_value("not json")


DEPTHS: list[tuple[FieldValue, int]] = [
    (1, 0),
    ("text", 0),
    ([], 1),
    ({}, 1),
    ([[], {"a": [1]}], 3),
    ({"a": {"b": "c"}}, 2),
]


@pytest.mark.parametrize("value, deep", DEPTHS)
def test_depth_counts_lists_and_objects(value: FieldValue, deep: int) -> None:
    assert depth(value) == deep


def test_lone_surrogates_are_replaced_in_every_string_however_nested() -> None:
    assert without_lone_surrogates([1, ("\udc00", {"\ud800": None})]) == [
        1,
        ["\N{REPLACEMENT CHARACTER}", {"\N{REPLACEMENT CHARACTER}": None}],
    ]
