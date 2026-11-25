from pathlib import Path
from typing import cast, get_type_hints

import pytest

from sift.api.collections import AddCollection, Audience, ShowStatus
from sift.api.documents import GetDocuments, WantedDocument
from sift.api.embedding import EmbedCollections
from sift.api.errors import NotFoundError
from sift.api.links import FindMentions, FindSimilar, ShowLinks
from sift.api.search import SearchHybrid, SearchKeywords, SearchResult, SearchVectors
from sift.app.collections import add
from sift.app.embedding import embed
from sift.interfaces.acts import collections, indexing, jobs, links, searching
from sift.interfaces.acts.acting import Act, Acting
from sift.interfaces.acts.collections import GET, STATUS
from sift.interfaces.acts.links import LINKS
from sift.interfaces.acts.searching import MENTIONS, QUERY, SEARCH, SIMILAR, VSEARCH
from sift.interfaces.server.handled import HANDLED
from sift.settings import Settings
from tests.files import write
from tests.homes import collection, indexed, quiet
from tests.models import Hashed

ACTS: list[Act[object, object]] = [
    value
    for module in (collections, indexing, jobs, links, searching)
    for value in cast("dict[str, object]", vars(module)).values()
    if isinstance(value, Act)
]


def test_every_act_served_names_the_answer_its_run_gives() -> None:
    assert sorted(act.name for act in ACTS) == sorted(HANDLED)
    assert [act.name for act in ACTS if act.answer != get_type_hints(act.run)["return"]] == []


@pytest.fixture
def agents(tmp_path: Path) -> Acting:
    home = tmp_path / "home"
    collection(tmp_path, home, "open", {"Plan.md": "Note: the plan of Kestrel.\n"})
    write(tmp_path / "hidden" / "Secret.md", "Note: Kestrel is the secret project.\n")
    add(home, AddCollection(str(tmp_path / "hidden"), agents=False))
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    settings = Settings(home=home, user=None, cwd=tmp_path)
    return Acting(settings, quiet, float, Hashed, Audience.AGENTS)


def _hits(result: object) -> list[str]:
    assert isinstance(result, SearchResult)
    return [hit.id for hit in result.hits]


def test_every_act_an_agent_reaches_keeps_to_the_collections_it_may_see(agents: Acting) -> None:
    searched = (
        SEARCH.run(agents, SearchKeywords("Kestrel")),
        VSEARCH.run(agents, SearchVectors("Kestrel")),
        QUERY.run(agents, SearchHybrid("Kestrel")),
    )
    assert [_hits(each) for each in searched] == [["open/Plan.md"]] * 3
    assert [one.name for one in STATUS.run(agents, ShowStatus()).collections] == ["open"]
    assert GET.run(agents, GetDocuments((WantedDocument("hidden/Secret.md"),))).found == ()
    hidden = "hidden/Secret.md"
    asked_of = (
        lambda: LINKS.run(agents, ShowLinks(hidden)),
        lambda: SIMILAR.run(agents, FindSimilar(hidden)),
        lambda: MENTIONS.run(agents, FindMentions(hidden)),
    )
    for each in asked_of:
        with pytest.raises(NotFoundError):
            each()
