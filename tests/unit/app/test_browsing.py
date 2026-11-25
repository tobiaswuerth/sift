from pathlib import Path

import pytest

from sift.api.browsing import Browsed, BrowseFolder, ListedDocument
from sift.api.collections import AddCollection, Audience
from sift.api.errors import InvalidRequestError, NotFoundError
from sift.app.browsing import browse
from sift.app.collections import add
from sift.app.documents import source_of
from tests.files import write
from tests.homes import collection, indexed

NOTES = {
    "a.md": "# A\n",
    "2024/b.md": "# B\n",
    "2024/c.md": "# C\n",
    "2024/deep/d.md": "# D\n",
    "2024-old/e.md": "# E\n",
    "z/[x]/f.md": "# F\n",
}


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    collection(tmp_path, home, "notes", NOTES)
    write(tmp_path / "hidden" / "s.md", "# S\n")
    add(home, AddCollection(str(tmp_path / "hidden"), agents=False))
    indexed(home)
    return home


def titles(browsed: Browsed) -> list[str]:
    return [each.title for each in browsed.documents]


def test_the_top_of_a_collection_shows_its_folders_and_every_document_by_path(home: Path) -> None:
    browsed = browse(home, BrowseFolder("notes"))
    assert (browsed.folders, titles(browsed), browsed.more) == (("2024", "2024-old", "z"), list("EBCDAF"), False)
    assert browsed.documents[0] == ListedDocument("notes/2024-old/e.md", "E")


@pytest.mark.parametrize("written", ["2024", "2024/", "/2024/", "2024//"])
def test_a_folder_is_a_folder_however_written(home: Path, written: str) -> None:
    browsed = browse(home, BrowseFolder("notes", written))
    assert (browsed.folder, browsed.folders, titles(browsed)) == ("2024/", ("deep",), ["B", "C", "D"])


def test_a_folder_named_with_brackets_is_browsed_as_named(home: Path) -> None:
    assert titles(browse(home, BrowseFolder("notes", "z/[x]"))) == ["F"]


def test_documents_past_k_are_one_more_page_away(home: Path) -> None:
    browsed = browse(home, BrowseFolder("notes", k=2))
    assert (titles(browsed), browsed.more) == (["E", "B"], True)


def test_a_folder_that_holds_nothing_shows_nothing(home: Path) -> None:
    assert browse(home, BrowseFolder("notes", "none")) == Browsed("notes", "none/", (), (), more=False)


def test_a_collection_hidden_from_agents_is_unknown_to_them(home: Path) -> None:
    assert titles(browse(home, BrowseFolder("hidden"))) == ["S"]
    with pytest.raises(NotFoundError):
        browse(home, BrowseFolder("hidden"), audience=Audience.AGENTS)


def test_a_folder_climbing_out_is_refused(home: Path) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        browse(home, BrowseFolder("notes", "2024/../.."))
    assert caught.value.field == "folder"


def test_a_document_s_source_is_its_file_in_its_collection(home: Path, tmp_path: Path) -> None:
    assert source_of(home, "notes/2024/b.md") == tmp_path / "notes" / "2024" / "b.md"
    assert source_of(home, str(tmp_path / "notes" / "2024" / "b.md")) == tmp_path / "notes" / "2024" / "b.md"
    with pytest.raises(NotFoundError):
        source_of(home, "notes/2024/none.md")
    with pytest.raises(NotFoundError):
        source_of(home, "hidden/s.md", audience=Audience.AGENTS)
