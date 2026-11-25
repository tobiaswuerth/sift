import json
from collections.abc import Iterator
from typing import SupportsIndex, final, override

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.api.fields import Fields, FieldValue
from sift.domain.text.front_matter import PARSE, TAKES, Brackets, Kind, entries, front_matter, read_on, value
from sift.domain.text.json_values import MOST_DEPTH, depth


def fields(text: str) -> Fields:
    head = front_matter(text)
    assert head is not None
    return head.fields


def field(written: str) -> FieldValue:
    return fields(f"---\nkey: {written}\n---\n")["key"]


def test_front_matter_holds_fields_and_ends_where_the_body_starts() -> None:
    text = '---\ntitle: "A title"\ntags: ["a", "b"]\nyear: 2024\nnote: plain\nq: \'x\'\n  nested: no\n---\n# H\n'
    head = front_matter(text)
    assert head is not None
    assert head.fields == {"title": "A title", "tags": ["a", "b"], "year": 2024, "note": "plain", "q": "x"}
    assert text[head.end :] == "# H\n"


def test_a_value_of_several_lines_below_a_block_scalar_mark_is_its_text() -> None:
    text = (
        "---\n"
        "description: |\n"
        "  Zebrafish husbandry,\n"
        "    indented further.\n"
        "\n"
        "  A second paragraph.\n"
        "summary: >-\n"
        "  folded onto\n"
        "  one line\n"
        "\n"
        "  and a second\n"
        "empty: |\n"
        "tags: [x]\n"
        "kept: >+ # a comment\n"
        "\tby a tab\n"
        "---\n"
        "the body\n"
    )
    assert fields(text) == {
        "description": "Zebrafish husbandry,\n  indented further.\n\nA second paragraph.",
        "summary": "folded onto one line\nand a second",
        "empty": "",
        "tags": ["x"],
        "kept": "by a tab",
    }
    assert fields("---\na: | b\nc: >\nd: 1\n---\n") == {"a": "| b", "c": "", "d": 1}
    assert fields("---\na: |\n  first\na: last\n---\n") == {"a": "last"}


def test_json_constants_sqlite_cannot_store_stay_text() -> None:
    assert fields("---\na: NaN\nb: [1, Infinity]\n---\n") == {"a": "NaN", "b": [1, "Infinity"]}


def test_windows_line_endings() -> None:
    text = "---\r\ntitle: T\r\n---\r\nbody\r\n"
    head = front_matter(text)
    assert head is not None
    assert (head.fields, text[head.end :]) == ({"title": "T"}, "body\r\n")


@pytest.mark.parametrize(
    "written, parsed",
    [
        ("[Hans Mueller, H. Müller]", ["Hans Mueller", "H. Müller"]),
        ('["a", "b"]', ["a", "b"]),
        ("[a, 'b, c', \"d\", O'Brien]", ["a", "b, c", "d", "O'Brien"]),
        ("[1, two, 3.5, true]", [1, "two", 3.5, True]),
        ("[a, [b, c]]", ["a", ["b", "c"]]),
        ("[a, b, ]", ["a", "b"]),
        ("[ ]", []),
        ("[[Acme AG]]", "[[Acme AG]]"),
        ("[[Acme AG|Acme]]", "[[Acme AG|Acme]]"),
        ("[[[Anna Keller]], [[Peter Frei|Peter]]]", ["[[Anna Keller]]", "[[Peter Frei|Peter]]"]),
        ('["[[Anna Keller]]", "[[Peter Frei]]"]', ["[[Anna Keller]]", "[[Peter Frei]]"]),
        ("[[A]], [[B]]", "[[A]], [[B]]"),
        ("[Draft] My note", "[Draft] My note"),
        ("[unclosed, list", "[unclosed, list"),
        ('["unclosed, quote]', '["unclosed, quote]'),
        ("[a]]", "[a]]"),
    ],
)
def test_yaml_flow_lists_and_single_wikilinks(written: str, parsed: FieldValue) -> None:
    assert field(written) == parsed


def test_values_sqlite_cannot_store_or_python_cannot_nest_stay_as_written() -> None:
    deep = "[" * 5000 + "]" * 5000
    assert fields(f'---\ntitle: "a \\ud800 b"\nx: {deep}\n---\n') == {"title": "a \ufffd b", "x": deep}
    tags = fields('---\ntags: ["\\udc00", {"k\\ud800": ["v\\ud83d"]}, 1]\n---\n')["tags"]
    assert tags == ["\ufffd", {"k\ufffd": ["v\ufffd"]}, 1]
    assert field('"\\ud83d\\ude00"') == "\U0001f600"
    nested = "[" * 1100 + "1" + "]" * 1100
    assert fields(f"---\nx: {nested}\nsize: 1e400\n---\n") == {"x": nested, "size": "1e400"}
    deep = "[" * 999 + "1" + "]" * 999
    parsed = fields(f"---\ntags:\n  - {deep}\n  - fine\nk: [x, {deep}]\nok: [[1]]\n---\n")
    assert parsed == {"tags": [deep, "fine"], "k": f"[x, {deep}]", "ok": [[1]]}
    assert depth(dict(parsed)) <= MOST_DEPTH + 1


def test_a_list_of_many_items_is_read_whole() -> None:
    items = 20_000
    block = "tags:\n" + "".join(f"- item{i}\n" for i in range(items))
    assert fields(f"---\n{block}---\nbody\n")["tags"] == [f"item{i}" for i in range(items)]


def test_a_list_in_brackets_over_many_lines_is_read_whole() -> None:
    items = 20_000
    block = "tags: [\n" + "".join(f"  item{i},\n" for i in range(items)) + "]"
    assert fields(f"---\n{block}\n---\n")["tags"] == [f"item{i}" for i in range(items)]


def looked_at(looked: list[int]) -> type[str]:
    @final
    class Text(str):
        __slots__ = ()

        @override
        def __getitem__(self, key: SupportsIndex | slice, /) -> str:
            return Text(super().__getitem__(key))

        @override
        def strip(self, chars: str | None = None, /) -> str:
            return Text(super().strip(chars))

        @override
        def __iter__(self) -> Iterator[str]:
            looked.append(len(self))
            return super().__iter__()

    return Text


def test_each_line_of_a_list_in_brackets_is_looked_at_once_for_the_bracket_that_closes_it() -> None:
    looked: list[int] = []
    line = looked_at(looked)
    items = 2000
    lines = [line("tags: ["), *(line(f"  item{i},") for i in range(items)), line("]"), line("year: 2024")]
    found = entries(lines).found
    assert [(entry.kind, len(below)) for entry, below in found.values()] == [(Kind.FLOW, items + 1), (Kind.PLAIN, 0)]
    assert 0 < sum(looked) <= sum(map(len, lines))


def test_a_list_nested_deeper_than_lists_in_brackets_go_is_kept_as_written_after_few_looks() -> None:
    looked: list[int] = []
    nested = "[a, " * 2000 + "b" + "]" * 2000
    assert value(looked_at(looked)(nested)) == nested
    assert sum(looked) <= 3 * len(nested)
    assert value("[a, [b, [c]], 'd']") == ["a", ["b", ["c"]], "d"]


@given(st.text(st.sampled_from("ab[]{},'\" \t"), max_size=30), st.text(st.sampled_from("ab[]{},'\" \t"), max_size=30))
def test_reading_brackets_on_from_a_state_is_reading_the_whole_at_once(first: str, second: str) -> None:
    assert read_on(second, read_on(first, Brackets())) == read_on(first + second, Brackets())


@pytest.mark.parametrize(
    "text",
    [
        "---\ntitle: x\nno end\n",
        "---\nSome intro words xyzzy\n---\nbody text\n",
        "---\ntitle: x\nxyzzy, a line of prose\n---\nbody text\n",
    ],
)
def test_what_is_no_front_matter(text: str) -> None:
    assert front_matter(text) is None


@pytest.mark.parametrize(
    "block, tags",
    [
        ("tags:\n-\n- a\n", ["a"]),
        ("tags: [\n  a,\n  b\n]\n", ["a", "b"]),
        ("tags: [a,\n\n  'b, c',\n  [d, O'Brien]]\n", ["a", "b, c", ["d", "O'Brien"]]),
        ("tags: {\n  a: 1\n}\n", "{ a: 1 }"),
    ],
)
def test_lines_yaml_allows_at_the_margin_keep_it_front_matter(block: str, tags: FieldValue) -> None:
    assert fields(f"---\ntitle: T\n{block}---\nbody\n") == {"title": "T", "tags": tags}


@pytest.mark.parametrize(
    "block, parsed",
    [
        ("title: [WIP My note\ntags: [a, b]\n", {"title": "[WIP My note", "tags": ["a", "b"]}),
        ("tags: [a,\nyear: 2024\n", {"tags": "[a,", "year": 2024}),
        ("tags: [\n  a,\n", {"tags": "["}),
        ("tags: [\n  a\n  ]\nyear: 2024\n", {"tags": ["a"], "year": 2024}),
    ],
)
def test_a_list_begun_and_never_closed_takes_no_other_field_with_it(block: str, parsed: Fields) -> None:
    assert fields(f"---\n{block}---\nbody\n") == parsed


def test_an_empty_front_matter_holds_no_fields() -> None:
    text = "---\n---\n# Title\nbody\n"
    head = front_matter(text)
    assert head is not None
    assert (head.fields, head.inner, text[head.end :]) == ({}, "", "# Title\nbody\n")


def test_items_below_a_key_and_comments_as_obsidian_jekyll_and_hugo_write_them() -> None:
    text = '---\n# a comment\ntags:\n  - painting\n  - "robots"\n- 2024\nauthors: ["A"]\nempty:\n---\nbody\n'
    assert fields(text) == {"tags": ["painting", "robots", 2024], "authors": ["A"], "empty": None}


def test_the_lines_between_the_rules_are_known_with_where_they_start() -> None:
    text = "---\na: 1\nb: 2\n---\nbody\n"
    head = front_matter(text)
    assert head is not None
    assert (head.inner, text[head.inner_start : head.inner_start + len(head.inner)]) == ("a: 1\nb: 2", "a: 1\nb: 2")


TRICKY = [
    "a: b",
    'say "hi"',
    "it's",
    "[Draft] My note",
    "[a, b]",
    "[[Acme AG]] and more",
    "# not a comment",
    "  padded  ",
    "C:\\Users\\Hans",
    '{"a": 1}',
    "Zürich \N{CHECK MARK} \N{CJK UNIFIED IDEOGRAPH-65E5}",
    "two\nlines",
    "line\N{LINE SEPARATOR}separator",
    "next\N{NEXT LINE}line",
    "para\N{PARAGRAPH SEPARATOR}graph",
    "---",
    "",
    "123",
    "true",
    "null",
]


@pytest.mark.parametrize("written", TRICKY)
def test_a_json_string_reads_back_as_written(written: str) -> None:
    for encoded in (json.dumps(written), json.dumps(written, ensure_ascii=False)):
        assert fields(f"---\nkey: {encoded}\ntitle: {encoded}\n---\n") == {"key": written, "title": written}


@pytest.mark.parametrize("written", [TRICKY, {"names": TRICKY, "n": -2.5e10, "ok": False, "none": None}, 0, True, None])
def test_any_json_value_reads_as_json(written: FieldValue) -> None:
    assert field(json.dumps(written)) == written


@given(st.text(st.sampled_from("ab :#-[]{}\"'\\|>!&*,\t\nä\N{CHECK MARK}\N{LINE SEPARATOR}"), max_size=40))
def test_every_json_string_reads_back_as_written(written: str) -> None:
    assert field(json.dumps(written)) == written


def test_every_kind_of_entry_knows_which_lines_it_takes_and_how_it_reads_them() -> None:
    assert set(TAKES) == set(PARSE) == set(Kind)


@pytest.mark.parametrize(
    "text, fields",
    [
        ("---\n# owner: Bob\nx: 1\n---\n", {"x": 1}),
        ("---\nrelated:\n  - see: [[Alice]]\n---\n", {"related": ["see: [[Alice]]"]}),
        ("---\na: \"x\nb: 'y\n---\n", {"a": "x", "b": "y"}),
    ],
    ids=["a comment line", "a list item holding a key", "values quoted on one side"],
)
def test_comments_items_holding_keys_and_half_quoted_values_read_as_they_did(text: str, fields: Fields) -> None:
    found = front_matter(text)
    assert found is not None
    assert found.fields == fields
