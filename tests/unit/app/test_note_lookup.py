from collections.abc import Callable
from pathlib import Path

import pytest

from sift.api.collections import AddCollection
from sift.api.errors import InvalidRequestError, NotFoundError
from sift.api.filters import written
from sift.api.indexing import IndexCollections
from sift.api.links import FindMentions, FindSimilar, ShowLinks
from sift.api.search import SearchKeywords
from sift.app.collections import add
from sift.app.indexing import index
from sift.app.links import links
from sift.app.mentions import mentions
from sift.app.search import search
from sift.app.similar import similar


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    corpus = {
        "A/Acme.md": "Note: widgets made by Acme.\n",
        "A/Hub.md": '---\norg: "[[Acme]]"\n---\nNote: widgets, see [[Acme]].\n',
        "A/x/Dup.md": "Note: one.\n",
        "B/blinker.md": '---\norg: "[[Acme]]"\n---\nNote: widgets, see [[Acme]].\n',
        "B/other.md": "Note: Acme ships widgets.\n",
        "B/Dup.md": "Note: two.\n",
    }
    for path, text in corpus.items():
        write(tmp_path / path, text)
    for name in ("A", "B"):
        add(home, AddCollection(str(tmp_path / name)))
    index(home, IndexCollections(), lambda _: None)
    return home


def ids(home: Path, request: SearchKeywords) -> list[str]:
    return sorted(hit.id for hit in search(home, request).hits)


def test_a_note_is_the_same_note_whichever_collections_are_searched(home: Path) -> None:
    assert ids(home, SearchKeywords("widgets", links_to=("Acme",))) == ["A/Hub.md"]
    with pytest.raises(InvalidRequestError, match=r"'Acme' is a note of A, and links stay within their collection"):
        search(home, SearchKeywords("widgets", collection="B", links_to=("Acme",)))


def test_a_store_ruled_out_by_the_links_is_counted_apart_from_those_searched(home: Path) -> None:
    found = search(home, SearchKeywords("widgets", links_to=("Acme",)))
    assert (found.searched, found.ruled_out) == (1, 1)


def test_a_link_filter_and_links_to_agree(home: Path) -> None:
    by_filter = ids(home, SearchKeywords("widgets", filters=(written("org=[[Acme]]"),)))
    assert by_filter == ids(home, SearchKeywords("widgets", links_to=("Acme",))) == ["A/Hub.md"]
    for asked in (
        SearchKeywords("widgets", links_to=("Dup",)),
        SearchKeywords("widgets", filters=(written("org=[[Dup]]"),)),
    ):
        with pytest.raises(InvalidRequestError, match="names several notes"):
            search(home, asked)


def test_mentions_leave_out_documents_of_other_collections_linking_the_note(home: Path) -> None:
    assert [hit.id for hit in mentions(home, FindMentions("A/Acme.md")).hits] == ["B/other.md"]


def _mentions(home: Path) -> object:
    return mentions(home, FindMentions("Nobody"))


def _similar(home: Path) -> object:
    return similar(home, FindSimilar("Nobody"))


def _links(home: Path) -> object:
    return links(home, ShowLinks("Nobody"))


def _links_to(home: Path) -> object:
    return search(home, SearchKeywords("widgets", links_to=("Nobody",)))


def _linked_from(home: Path) -> object:
    return search(home, SearchKeywords("widgets", linked_from=("Nobody",)))


@pytest.mark.parametrize("asked", [_mentions, _similar, _links, _links_to, _linked_from])
def test_a_name_no_note_answers_to_is_said_alike_everywhere(home: Path, asked: Callable[[Path], object]) -> None:
    with pytest.raises(NotFoundError, match=r"^no note or document 'Nobody'$"):
        asked(home)
