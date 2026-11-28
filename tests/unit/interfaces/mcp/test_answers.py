from typing import cast

import pytest

from sift.api.errors import InternalError
from sift.api.search import Hit, SearchResult
from sift.interfaces.acts.acting import answered
from sift.interfaces.mcp.answers import hits_of


def test_a_hit_held_by_several_collections_names_its_copies() -> None:
    hit = Hit("a/x.md", "a", "x.md", "X", 1, 2, 0.5, None, also=("b/x.md",))
    shown, held = hits_of(SearchResult((hit,), 2, (), scored_by="keywords"))
    assert (shown["hits"], held) == (
        [{"id": "a/x.md", "title": "X", "first_line": 1, "last_line": 2, "score": 0.5, "also": ["b/x.md"]}],
        0,
    )


def test_a_hit_keeps_its_text_while_the_answer_has_room_for_it_and_the_rest_are_counted_in_a_note() -> None:
    hits = tuple(Hit(f"a/{n}.md", "a", f"{n}.md", "X", 1, 2, 0.5, "four" * n) for n in (1, 2, 3))
    shown, held = hits_of(SearchResult(hits, 1, (), scored_by="keywords"), room=12)
    found = cast("list[dict[str, object]]", shown["hits"])
    assert ([hit.get("text") for hit in found], held) == (["four", "fourfour", None], 12)
    assert shown["notes"] == [
        "the text of 1 hit is left out, as this answer's text is at its 100,000 characters: get them by id and lines"
    ]


def test_an_answer_of_another_kind_is_a_bug_said_to_the_agent() -> None:
    with pytest.raises(InternalError, match="a str came back for a SearchResult"):
        answered("hits", SearchResult)
