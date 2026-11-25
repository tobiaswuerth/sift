import pytest

from sift.api.errors import InternalError
from sift.api.search import Hit, SearchResult
from sift.interfaces.acts.acting import answered
from sift.interfaces.mcp.answers import hits_of


def test_a_hit_held_by_several_collections_names_its_copies() -> None:
    hit = Hit("a/x.md", "a", "x.md", "X", 1, 2, 0.5, None, also=("b/x.md",))
    shown = hits_of(SearchResult((hit,), 2, (), scored_by="keywords"))
    assert shown["hits"] == [
        {"id": "a/x.md", "title": "X", "first_line": 1, "last_line": 2, "score": 0.5, "also": ["b/x.md"]}
    ]


def test_an_answer_of_another_kind_is_a_bug_said_to_the_agent() -> None:
    with pytest.raises(InternalError, match="a str came back for a SearchResult"):
        answered("hits", SearchResult)
