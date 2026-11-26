from pathlib import Path
from typing import cast

import pytest

from sift.api.errors import InvalidRequestError
from sift.interfaces.mcp.answers import MOST_TEXT
from tests.agent_tools import Direct, Object, called
from tests.files import write
from tests.homes import indexed


def test_get_reads_documents_and_lines_and_says_what_it_could_not(direct: Direct) -> None:
    answer = called(
        direct,
        "get",
        {"documents": [{"id": "open/Plan.md", "first_line": 6, "last_line": 6}, {"id": "hidden/Secret.md"}]},
    )
    assert answer["documents"] == [
        {"id": "open/Plan.md", "title": "Plan", "text": "Note: the plan of Kestrel names [[Secret]].\n"}
    ]
    (refusal,) = cast("list[str]", answer["not_read"])
    assert refusal.startswith("hidden/Secret.md: no note or document 'hidden/Secret.md'; fix: ")


def test_get_cuts_a_text_at_a_line_where_one_call_is_full_and_says_how_to_get_the_rest(
    direct: Direct, tmp_path: Path
) -> None:
    for name in ("a", "b"):
        write(tmp_path / "open" / f"{name}.md", f"# {name}\n\n" + "word " * (MOST_TEXT // 8) + "\nthe end\n")
    indexed(direct.acting.settings.home)
    asked = ["open/a.md", {"id": "open/b.md"}, {"id": "open/b.md", "first_line": 3}, "open/Plan.md"]
    whole, cut, later, short = cast("list[Object]", called(direct, "get", {"documents": asked})["documents"])
    assert (set(whole), cast("str", whole["text"]).endswith("the end\n")) == ({"id", "title", "text"}, True)
    assert (set(short), cast("str", short["text"]).startswith("---\ncreated")) == ({"id", "title", "text"}, True)
    full = f"this answer's text is at its {MOST_TEXT:,} characters"
    assert cut == {
        "id": "open/b.md",
        "title": "b",
        "text": "# b\n\n",
        "first_line": 1,
        "last_line": 2,
        "lines": 4,
        "left_out": f"lines 3 to 4 are left out, as {full}: get them with first_line 3",
    }
    assert later == {
        "id": "open/b.md",
        "title": "b",
        "left_out": f"left out, as this answer's text is at its {MOST_TEXT:,} characters: get it in a call of its own",
    }
    rest = called(direct, "get", {"documents": [{"id": "open/b.md", "first_line": 3}]})
    (read,) = cast("list[Object]", rest["documents"])
    assert (set(read), cast("str", read["text"]).endswith("the end\n")) == ({"id", "title", "text"}, True)


def test_get_reads_a_document_longer_than_one_call_holds_in_calls_that_follow_each_other(
    direct: Direct, tmp_path: Path
) -> None:
    write(tmp_path / "open" / "long.md", "".join(f"line {n:05} of a long note\n" for n in range(1, 6_001)))
    indexed(direct.acting.settings.home)
    read, first = list[str](), 1
    while first:
        (one,) = cast(
            "list[Object]",
            called(direct, "get", {"documents": [{"id": "open/long.md", "first_line": first}]})["documents"],
        )
        read.append(cast("str", one["text"]))
        assert len(read[-1]) <= MOST_TEXT
        first = cast("int", one["last_line"]) + 1 if "left_out" in one else 0
    assert (len(read), "".join(read).count("\n"), read[0][:28]) == (2, 6_000, "line 00001 of a long note\nli")


def test_get_gives_the_start_of_a_line_longer_than_one_call_holds(direct: Direct, tmp_path: Path) -> None:
    write(tmp_path / "open" / "wide.txt", "first\n" + "word " * (MOST_TEXT // 4) + "\nlast\n")
    indexed(direct.acting.settings.home)
    (one,) = cast(
        "list[Object]", called(direct, "get", {"documents": [{"id": "open/wide.txt", "first_line": 2}]})["documents"]
    )
    assert (len(cast("str", one["text"])), one["first_line"], one["last_line"], one["lines"]) == (MOST_TEXT, 2, 2, 3)
    assert one["left_out"] == (
        f"line 2 goes on for 25,000 characters more than the {MOST_TEXT:,} an answer holds; they are left out, "
        "and the lines after it: get those with first_line 3"
    )


def test_get_refuses_a_document_that_is_neither_an_id_nor_an_object_saying_what_to_give(direct: Direct) -> None:
    with pytest.raises(InvalidRequestError) as refused:
        called(direct, "get", {"documents": [7]})
    assert (refused.value.field, refused.value.remedy) == (
        "documents[0]",
        'give each document as its id, or as {"id": "...", "first_line": 17, "last_line": 26} for some of its lines',
    )
    with pytest.raises(InvalidRequestError) as unlisted:
        called(direct, "get", {"documents": "open/Plan.md"})
    assert unlisted.value.field == "documents"


def test_get_shows_the_front_matter_asked_for(direct: Direct) -> None:
    answer = called(direct, "get", {"documents": [{"id": "open/Plan.md"}], "fields": ["created"]})
    (document,) = cast("list[Object]", answer["documents"])
    assert document["fields"] == {"created": "2026-01-02"}
