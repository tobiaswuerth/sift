from pathlib import Path

from sift.api.collections import AddCollection
from sift.api.errors import Problem
from sift.api.search import SearchKeywords
from sift.app.collections import add
from sift.app.search import search
from tests.files import write
from tests.homes import indexed

NESTED = "---\nauthor:\n  name: Zanzibarian\npeople:\n  - name: A\n    role: Wombatchief\n---\n# Note\n\nbody\n"


def test_a_search_finds_the_words_of_nested_front_matter(tmp_path: Path) -> None:
    write(tmp_path / "notes" / "n.md", NESTED)
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "notes")))
    indexed(home)
    for word in ("Zanzibarian", "Wombatchief"):
        assert [hit.id for hit in search(home, SearchKeywords(word)).hits] == ["notes/n.md"]


def test_a_field_to_sort_by_or_show_that_no_result_holds_is_said(tmp_path: Path) -> None:
    write(tmp_path / "notes" / "a.md", "---\nyear: 2024\n---\n# A\n\npigment\n")
    write(tmp_path / "notes" / "b.md", "---\nyear: 2023\n---\n# B\n\npigment pigment\n")
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "notes")))
    indexed(home)
    found = search(home, SearchKeywords("pigment", sort="-yeer", fields=("year", "tiltle", "*")))
    assert [hit.id for hit in found.hits] == [hit.id for hit in search(home, SearchKeywords("pigment")).hits]
    assert [(each.code, each.message, each.collection) for each in found.left_out] == [
        (Problem.FIELD_NOT_HELD, "no result holds 'yeer', the field to sort by: the results stand as ranked", None),
        (Problem.FIELD_NOT_HELD, "no result holds the field 'tiltle'", None),
    ]
    held = search(home, SearchKeywords("pigment", sort="year", fields=("year",)))
    assert (held.left_out, [hit.id for hit in held.hits]) == ((), ["notes/b.md", "notes/a.md"])
    assert search(home, SearchKeywords("nowhere", sort="yeer", fields=("tiltle",))).left_out == ()
