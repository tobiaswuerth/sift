from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from sift.api.collections import AddCollection, Audience, ShowStatus
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, NotFoundError, Problem
from sift.api.links import FindMentions, FindSimilar, ListUnresolved, ShowLinks
from sift.api.search import Options, SearchHybrid, SearchKeywords, SearchResult, SearchVectors
from sift.app.collections import add, status
from sift.app.documents import documents
from sift.app.embedding import embed
from sift.app.indexing import files_of, index_files
from sift.app.links import links, unresolved
from sift.app.mentions import mentions
from sift.app.query import query
from sift.app.search import search
from sift.app.searching import Watching
from sift.app.similar import similar
from sift.app.vsearch import vsearch
from tests.files import write
from tests.homes import asked, collection, indexed, quiet
from tests.models import Hashed

AGENTS = Audience.AGENTS
SEEN = "the collections are: open"


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    collection(tmp_path, home, "open", {"Plan.md": "Note: the plan of Kestrel names [[Secret]].\n"})
    write(tmp_path / "hidden" / "Secret.md", "Note: Kestrel is the secret project, see [[Plan]].\n")
    add(home, AddCollection(str(tmp_path / "hidden"), agents=False))
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    return home


def ids(result: SearchResult) -> list[str]:
    return sorted(hit.id for hit in result.hits)


def searched(home: Path, options: Options, audience: Audience) -> SearchResult:
    match options:
        case SearchHybrid():
            return query(home, options, Hashed(), watching=Watching(audience=audience))
        case SearchVectors():
            return vsearch(home, options, Hashed(), watching=Watching(audience=audience))
        case SearchKeywords():
            return search(home, options, Watching(audience=audience))
        case _:
            raise AssertionError(options)


SEARCHES = (SearchKeywords("Kestrel"), SearchVectors("Kestrel"), SearchHybrid("Kestrel"))


@pytest.mark.parametrize("asked_for", SEARCHES, ids=type)
def test_an_agent_searches_only_the_collections_it_may_see(home: Path, asked_for: Options) -> None:
    assert ids(searched(home, asked_for, Audience.PEOPLE)) == ["hidden/Secret.md", "open/Plan.md"]
    assert ids(searched(home, asked_for, AGENTS)) == ["open/Plan.md"]


@pytest.mark.parametrize("asked_for", SEARCHES, ids=type)
def test_a_hidden_collection_an_agent_names_is_unknown_and_its_remedy_names_the_visible_alone(
    home: Path, asked_for: Options
) -> None:
    with pytest.raises(NotFoundError) as caught:
        searched(home, replace(asked_for, collections=("hidden",)), AGENTS)
    assert (caught.value.code, caught.value.remedy) == (Problem.UNKNOWN_COLLECTION, SEEN)


def test_a_hidden_note_an_agent_filters_by_links_to_is_unknown(home: Path) -> None:
    assert ids(search(home, SearchKeywords("Kestrel", links_to=("hidden/Secret.md",)))) == []
    with pytest.raises(NotFoundError) as caught:
        search(home, SearchKeywords("Kestrel", linked_from=("hidden/Secret.md",)), Watching(audience=AGENTS))
    assert caught.value.code is Problem.UNKNOWN_NOTE


def test_a_hidden_document_an_agent_gets_is_unknown_by_id_and_by_path(home: Path, tmp_path: Path) -> None:
    full = str(tmp_path / "hidden" / "Secret.md")
    assert [one.id for one in documents(home, asked("hidden/Secret.md", full)).found] == ["hidden/Secret.md"] * 2
    got = documents(home, asked("hidden/Secret.md", full, "open/Plan.md"), audience=AGENTS)
    assert [one.id for one in got.found] == ["open/Plan.md"]
    assert [notice.code for notice in got.not_read] == [Problem.UNKNOWN_DOCUMENT] * 2


def _links(home: Path, note: str) -> object:
    return links(home, ShowLinks(note), audience=AGENTS)


def _similar(home: Path, note: str) -> object:
    return similar(home, FindSimilar(note), watching=Watching(audience=AGENTS))


def _mentions(home: Path, note: str) -> object:
    return mentions(home, FindMentions(note), Watching(audience=AGENTS))


@pytest.mark.parametrize("asked_of", [_links, _similar, _mentions], ids=["links", "similar", "mentions"])
def test_a_hidden_note_an_agent_asks_about_is_unknown_by_id_and_by_name(
    home: Path, asked_of: Callable[[Path, str], object]
) -> None:
    for note in ("hidden/Secret.md", "Secret"):
        with pytest.raises(NotFoundError) as caught:
            asked_of(home, note)
        assert caught.value.code is Problem.UNKNOWN_NOTE


def test_an_agent_sees_the_status_of_the_collections_it_may_see(home: Path) -> None:
    assert [one.name for one in status(home, ShowStatus()).collections] == ["hidden", "open"]
    assert [one.name for one in status(home, ShowStatus(), audience=AGENTS).collections] == ["open"]
    with pytest.raises(NotFoundError) as caught:
        status(home, ShowStatus(("hidden",)), audience=AGENTS)
    assert caught.value.remedy == SEEN


def test_unresolved_links_are_listed_for_people(home: Path) -> None:
    assert sorted(each.target for each in unresolved(home, ListUnresolved()).targets) == ["Plan", "Secret"]


def test_an_agent_may_index_files_of_the_collections_it_may_see_alone(home: Path, tmp_path: Path) -> None:
    secret, plan = str(tmp_path / "hidden" / "Secret.md"), str(tmp_path / "open" / "Plan.md")
    with pytest.raises(InvalidRequestError) as caught:
        files_of(home, (plan, secret), AGENTS)
    assert (caught.value.message, caught.value.field) == (f"{secret} is in no collection", "files")
    grouped = files_of(home, (plan, "open/Plan.md"), AGENTS)
    assert {one.name: paths for one, paths in grouped.items()} == {"open": ("Plan.md",)}
    write(tmp_path / "open" / "Plan.md", "Note: a new plan.\n")
    (indexed,) = index_files(home, grouped, quiet).collections
    assert [(each.id, each.state) for each in indexed.named] == [("open/Plan.md", "changed")]
