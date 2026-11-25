from collections.abc import Callable
from pathlib import Path

import pytest

from sift.api.collections import AddCollection, DeactivateCollections, RemoveCollection, ShowStatus
from sift.api.errors import InvalidRequestError, NotFoundError, StoreUnavailableError
from sift.api.filters import Filter, Operator
from sift.api.indexing import IndexCollections
from sift.api.links import FindMentions, FindSimilar, ShowLinks
from sift.api.search import SearchKeywords
from sift.app.collections import add, deactivate, remove, status
from sift.app.documents import documents
from sift.app.indexing import index
from sift.app.links import links
from sift.app.mentions import mentions
from sift.app.search import search
from sift.app.similar import similar
from sift.storage.catalog import CATALOG
from tests.files import write
from tests.homes import asked

type Asking = Callable[[Path], object]


@pytest.fixture
def home(tmp_path: Path) -> Path:
    root, home = tmp_path / "notes", tmp_path / "home"
    for path in ("a/b/c.md", "x/b/c.md", "rl.md", "i.md"):
        write(root / path, f"# {path}\n")
    write(tmp_path / "other" / "o.md", "# Other\n")
    add(home, AddCollection(str(root)))
    add(home, AddCollection(str(tmp_path / "other")))
    index(home, IndexCollections(), lambda _: None)
    deactivate(home, DeactivateCollections(("other",)))
    return home


REFUSALS: dict[str, tuple[Asking, str]] = {
    "a query the grammar refuses": (lambda home: search(home, SearchKeywords('"halo')), "query"),
    "an unknown collection searched": (
        lambda home: search(home, SearchKeywords("halo", collections=("nope",))),
        "collections",
    ),
    "an inactive collection searched": (
        lambda home: search(home, SearchKeywords("halo", collections=("other",))),
        "collections",
    ),
    "an unknown collection indexed": (lambda home: index(home, IndexCollections(("nope",)), lambda _: None), "names"),
    "an unknown collection shown": (lambda home: status(home, ShowStatus(("nope",))), "names"),
    "an unknown collection removed": (lambda home: remove(home, RemoveCollection("nope")), "name"),
    "an ambiguous id to find like": (lambda home: similar(home, FindSimilar("notes/c.md")), "id"),
    "an ambiguous id to show the links of": (lambda home: links(home, ShowLinks("notes/c.md")), "id"),
    "an unknown note to find the mentions of": (lambda home: mentions(home, FindMentions("notes/none.md")), "id"),
    "an unknown note searched for links to": (
        lambda home: search(home, SearchKeywords("halo", links_to=("notes/c.md",))),
        "links_to",
    ),
    "lines asked the wrong way round": (
        lambda home: documents(home, asked("notes/rl.md", "notes/i.md", lines=(3, 2))),
        "documents[0].last_line",
    ),
    "a filter whose value reads as a list": (
        lambda home: search(home, SearchKeywords("halo", filters=(Filter("tags", Operator.ANY, ("[a, b]",)),))),
        "filters",
    ),
}


@pytest.mark.parametrize("asking, field", REFUSALS.values(), ids=REFUSALS.keys())
def test_a_refusal_names_the_field_at_fault(home: Path, asking: Asking, field: str) -> None:
    with pytest.raises((InvalidRequestError, NotFoundError)) as refused:
        asking(home)
    assert refused.value.field == field


def test_a_refusal_not_of_a_request_s_field_names_none(home: Path) -> None:
    (home / CATALOG).write_bytes(b"no catalog")
    with pytest.raises(StoreUnavailableError) as refused:
        search(home, SearchKeywords("halo", collections=("notes",)))
    assert refused.value.field is None
