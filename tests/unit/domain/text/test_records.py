import json
import unicodedata

import pytest

from sift.api.fields import FieldValue
from sift.domain.text.notes import Document
from sift.domain.text.records import (
    NO_CHOICE,
    NOT_A_KEY,
    Chosen,
    SkippedLine,
    record_of,
    used_twice,
)


def read(value: object, chosen: Chosen = NO_CHOICE, line: int = 1) -> Document:
    found = record_of(json.dumps(value), line, chosen)
    assert isinstance(found, Document)
    return found


def test_the_text_is_title_and_text_field_and_every_other_field_front_matter() -> None:
    paper = {"id": "2101.00001", "title": "Brush stroke planning", "abstract": "A robot plans.", "year": 2021}
    found = read(paper)
    assert (found.key, found.title, found.text) == (
        "2101.00001",
        "Brush stroke planning",
        "# Brush stroke planning\n\nA robot plans.",
    )
    assert found.fields == {"id": "2101.00001", "title": "Brush stroke planning", "year": 2021}
    assert [(each.start, each.length) for each in found.chunks] == [(0, len(found.text))]
    assert found.links == ()


@pytest.mark.parametrize(
    "value, title, text",
    [
        ({"id": 1, "title": "T", "text": "body", "abstract": "not this"}, "T", "# T\n\nbody"),
        ({"_id": "a", "name": "N", "content": "body"}, "N", "# N\n\nbody"),
        ({"id": 1, "headline": "H", "body": ["one", "two"]}, "H", "# H\n\none\n\ntwo"),
        ({"id": 1, "subject": "S", "description": "body"}, "S", "# S\n\nbody"),
        ({"id": 1, "summary": "# Own heading\n\nbody"}, "Own heading", "# Own heading\n\nbody"),
        (
            {"id": 1, "question": "short", "answer": "the longest string is the text"},
            "1",
            "the longest string is the text",
        ),
        (
            {"id": 1, "title": "Brush stroke\n planning", "text": "x"},
            "Brush stroke planning",
            "# Brush stroke planning\n\nx",
        ),
        (
            {"id": 1, "title": ["Brush stroke", "planning"], "text": "x"},
            "Brush stroke planning",
            "# Brush stroke planning\n\nx",
        ),
    ],
)
def test_fields_are_found_by_name(value: dict[str, FieldValue], title: str, text: str) -> None:
    found = read(value)
    assert (found.title, found.text) == (title, text)


def test_a_record_without_an_id_is_named_by_its_line_and_fields_may_be_chosen() -> None:
    assert read({"text": "first"}, line=7).key == "7"
    assert read({"id": "", "text": "x"}, line=8).key == "8"
    chosen = Chosen(id="key", title="q", text=("q", "a"))
    found = read({"key": "q1", "q": "Why?", "a": "Because.", "text": "not this"}, chosen)
    assert (found.key, found.title, found.text) == ("q1", "Why?", "# Why?\n\nWhy?\n\nBecause.")
    assert found.fields == {"key": "q1", "text": "not this"}


@pytest.mark.parametrize("value", [{"text": "x"}, {"doi": "", "text": "x"}, {"doi": None, "text": "x"}])
def test_a_record_lacking_the_id_field_chosen_is_skipped_as_having_no_id(value: dict[str, FieldValue]) -> None:
    assert record_of(json.dumps(value), 5, Chosen(id="doi")) == SkippedLine(5, "no id: it holds no doi")


def test_a_key_used_twice_says_whether_a_record_without_an_id_holds_it() -> None:
    first, explicit, third = (
        read({"text": "a"}, line=1),
        read({"id": 1, "text": "b"}, line=2),
        read({"text": "c"}, line=3),
    )
    advice = "give every record an id or none"
    assert [
        used_twice(explicit, first, NO_CHOICE),
        used_twice(third, None, NO_CHOICE),
        used_twice(explicit, explicit, NO_CHOICE),
        used_twice(explicit, None, NO_CHOICE),
    ] == [
        f"id 1 is also line 1's, which has no id: {advice}",
        f"it has no id, and id 3 is another record's: {advice}",
        "id 1 used twice",
        "id 1 used twice",
    ]


@pytest.mark.parametrize("key", [True, 1.5, [1], {"a": 1}, "a\nb", "a\x1bb", "\x85"])
def test_an_id_that_is_no_text_or_whole_number_skips_its_line(key: FieldValue) -> None:
    assert record_of(json.dumps({"id": key, "text": "x"}), 4, NO_CHOICE) == SkippedLine(4, NOT_A_KEY)


def test_a_whole_number_and_its_text_are_the_same_id() -> None:
    assert (
        read({"id": 7, "text": "x"}).key,
        read({"id": "7", "text": "x"}).key,
        read({"id": -2, "text": "x"}).key,
    ) == (
        "7",
        "7",
        "-2",
    )


@pytest.mark.parametrize(
    "line, why",
    [
        ("{not json", "not JSON"),
        ("[1, 2]", "not an object"),
        ('{"id": 1}', "no text"),
        ("[" * 100_000 + "]" * 100_000, "nested too deeply"),
        ('{"id": "4", "text": "x", "v": ' + "[" * 1100 + "1" + "]" * 1100 + "}", "nested too deeply"),
    ],
)
def test_a_line_without_a_record_says_why(line: str, why: str) -> None:
    assert record_of(line, 3, NO_CHOICE) == SkippedLine(3, why)


def test_what_sqlite_cannot_store_is_made_storable() -> None:
    found = record_of('{"id": "1", "text": "a \\ud83d b", "size": 1e999, "n": 2.5}', 1, NO_CHOICE)
    assert isinstance(found, Document)
    assert (found.text, found.fields) == ("a \ufffd b", {"id": "1", "size": "1e999", "n": 2.5})
    deep = record_of('{"id": "5", "text": "near", "v": ' + "[" * 998 + "1" + "]" * 998 + "}", 1, NO_CHOICE)
    assert isinstance(deep, Document)


def test_a_record_s_digest_is_its_line_s() -> None:
    one, other = read({"id": 1, "text": "a"}), read({"id": 1, "text": "b"})
    assert one.digest != other.digest
    assert read({"id": 1, "text": "a"}).digest == one.digest


FIELDS = {
    "id": "a",
    "title": "T",
    "abstract": "Z\u00fcrich body",
    "tags": ["x", "na\u00efve"],
    "n": {"k": [1, "z\u00fc"]},
}


def test_a_line_composed_already_gives_the_record_its_escaped_or_decomposed_line_gives() -> None:
    escaped, written = json.dumps(FIELDS), json.dumps(FIELDS, ensure_ascii=False)
    decomposed = unicodedata.normalize("NFD", written)
    assert "\\u" in escaped
    assert written != decomposed
    first, second, third = (record_of(line, 1, NO_CHOICE) for line in (written, escaped, decomposed))
    assert isinstance(first, Document)
    assert isinstance(second, Document)
    assert isinstance(third, Document)
    assert (
        first.fields
        == second.fields
        == third.fields
        == {"id": "a", "title": "T", "tags": FIELDS["tags"], "n": FIELDS["n"]}
    )
    assert (first.key, first.title, first.chunks) == (second.key, second.title, second.chunks)
    assert first.text == second.text == "# T\n\nZ\u00fcrich body"
