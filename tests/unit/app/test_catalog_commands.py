from pathlib import Path

import pytest

from sift.api.collections import (
    ActivateCollections,
    AddCollection,
    CollectionRemoved,
    CollectionsSwitched,
    DeactivateCollections,
    RemoveCollection,
)
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, NotFoundError, Problem, StoreUnavailableError
from sift.api.indexing import IndexCollections
from sift.api.links import ListUnresolved, ShowLinks
from sift.api.search import SearchHybrid, SearchKeywords, SearchVectors
from sift.app.collections import activate, add, deactivate, remove
from sift.app.documents import documents
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.app.links import links, unresolved
from sift.app.query import query
from sift.app.search import search
from sift.app.vsearch import vsearch
from sift.storage.catalog import known_collections
from sift.storage.layout import LAYOUT
from sift.storage.store import create_store
from sift.storage.store_files import delete_store
from sift.storage.writer import sole_writer
from tests.files import write
from tests.homes import asked, home_of, indexed, quiet
from tests.models import Hashed

NOTES = {
    "notes/robots.md": "# Robots\n\nA robot vacuum broke. See [[Gone]].\n",
    "diary/monday.md": "# Monday\n\nThe robot vacuum broke again.\n",
    "diary/tuesday.md": "# Tuesday\n\nNothing broke.\n",
}


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return home_of(tmp_path, NOTES)


def found(home: Path, query_text: str = "robot") -> list[str]:
    return sorted(hit.id for hit in search(home, SearchKeywords(query_text)).hits)


def test_a_collection_removed_takes_its_store_and_leaves_its_files(home: Path, tmp_path: Path) -> None:
    (diary,) = [one for one in known_collections(home) if one.name == "diary"]
    assert remove(home, RemoveCollection("diary")) == CollectionRemoved("diary", str(diary.store), 2, None)
    assert (tmp_path / "diary" / "monday.md").is_file()
    assert not list(diary.store.parent.glob(f"{diary.store.name}*"))
    assert [one.name for one in known_collections(home)] == ["notes"]
    assert found(home) == ["notes/robots.md"]
    with pytest.raises(NotFoundError, match="no collection is named 'diary'"):
        remove(home, RemoveCollection("diary"))


def test_a_collection_whose_store_is_missing_is_taken_out_of_the_catalog(home: Path) -> None:
    (diary,) = [one for one in known_collections(home) if one.name == "diary"]
    diary.store.unlink()
    removed = remove(home, RemoveCollection("diary"))
    assert (removed.documents, removed.problem is not None) == (None, True)
    assert "missing" in str(removed.problem)
    assert [one.name for one in known_collections(home)] == ["notes"]


def test_a_collection_whose_store_is_damaged_is_removed_with_it(home: Path) -> None:
    (diary,) = [one for one in known_collections(home) if one.name == "diary"]
    diary.store.write_bytes(b"not a database, " * 1000)
    removed = remove(home, RemoveCollection("diary"))
    assert (removed.documents, removed.problem is not None) == (None, True)
    assert not diary.store.exists()


def test_a_collection_whose_store_is_of_an_older_layout_is_removed_with_its_documents_counted(home: Path) -> None:
    (diary,) = [one for one in known_collections(home) if one.name == "diary"]
    delete_store(diary.store)
    create_store(diary.store, diary.id, LAYOUT[:1])
    assert remove(home, RemoveCollection("diary")) == CollectionRemoved("diary", str(diary.store), 0, None)
    assert not diary.store.exists()


def test_a_store_open_elsewhere_when_its_collection_is_removed_is_deleted_by_the_next_index(home: Path) -> None:
    (diary,) = [one for one in known_collections(home) if one.name == "diary"]

    def held_open(file: Path) -> None:
        if file == diary.store:
            raise PermissionError(13, "The process cannot access the file", str(file))
        file.unlink()

    assert remove(home, RemoveCollection("diary"), held_open).left == (str(diary.store),)
    assert diary.store.exists()
    indexed(home)
    assert not diary.store.exists()


def test_a_collection_another_sift_writes_now_stays(home: Path) -> None:
    (diary,) = [one for one in known_collections(home) if one.name == "diary"]
    with sole_writer(diary.store, "sift index"), pytest.raises(StoreUnavailableError) as refused:
        remove(home, RemoveCollection("diary"))
    assert refused.value.code is Problem.BUSY
    assert [one.name for one in known_collections(home)] == ["diary", "notes"]
    assert found(home) == ["diary/monday.md", "notes/robots.md"]


def test_a_collection_added_after_one_removed_never_takes_its_id(home: Path, tmp_path: Path) -> None:
    removed = max(one.id for one in known_collections(home))
    remove(home, RemoveCollection("diary"))
    add(home, AddCollection(str(tmp_path / "diary"), name="again"))
    assert {one.name: one.id > removed for one in known_collections(home)} == {"again": True, "notes": False}


def test_collections_are_active_until_deactivated(home: Path) -> None:
    assert deactivate(home, DeactivateCollections(("diary", "notes", "Diary"))) == CollectionsSwitched(
        active=False, changed=("diary", "notes"), already=()
    )
    assert [one.active for one in known_collections(home)] == [False, False]
    assert deactivate(home, DeactivateCollections(("notes",))) == CollectionsSwitched(
        active=False, changed=(), already=("notes",)
    )
    assert activate(home, ActivateCollections(("notes",))) == CollectionsSwitched(
        active=True, changed=("notes",), already=()
    )
    assert {one.name: one.active for one in known_collections(home)} == {"diary": False, "notes": True}


def test_an_unknown_name_changes_nothing(home: Path) -> None:
    with pytest.raises(NotFoundError, match="'nope'"):
        deactivate(home, DeactivateCollections(("notes", "nope")))
    assert all(one.active for one in known_collections(home))
    with pytest.raises(InvalidRequestError, match="names holds 0"):
        ActivateCollections(())


def test_no_catalog_is_made_by_naming_a_collection(tmp_path: Path) -> None:
    home = tmp_path / "nowhere"
    for change in (
        lambda: remove(home, RemoveCollection("notes")),
        lambda: activate(home, ActivateCollections(("notes",))),
    ):
        with pytest.raises(NotFoundError):
            change()
    assert not home.exists()


def test_an_inactive_collection_is_passed_over_by_searches(home: Path) -> None:
    deactivate(home, DeactivateCollections(("diary",)))
    assert found(home) == ["notes/robots.md"]
    activate(home, ActivateCollections(("diary",)))
    assert found(home) == ["diary/monday.md", "notes/robots.md"]


def test_an_inactive_collection_named_is_refused_saying_how_to_search_it(home: Path) -> None:
    embed(home, EmbedCollections(), quiet, Hashed())
    deactivate(home, DeactivateCollections(("diary",)))
    for call in (
        lambda: search(home, SearchKeywords("robot", collections=("diary",))),
        lambda: vsearch(home, SearchVectors("robot", collections=("diary",)), Hashed()),
        lambda: query(home, SearchHybrid("robot", collections=("diary",)), Hashed()),
        lambda: unresolved(home, ListUnresolved(collections=("diary",))),
        lambda: links(home, ShowLinks("diary/monday.md")),
    ):
        with pytest.raises(InvalidRequestError, match="the collection 'diary' is inactive") as refused:
            call()
        assert (refused.value.code, refused.value.remedy) == (
            Problem.INACTIVE_COLLECTION,
            "activate it",
        )
    (unfound,) = documents(home, asked("diary/monday.md")).not_read
    assert (unfound.id, unfound.code) == ("diary/monday.md", Problem.INACTIVE_COLLECTION)


def test_with_every_collection_inactive_a_search_says_how_to_activate_one(home: Path) -> None:
    deactivate(home, DeactivateCollections(("diary", "notes")))
    for call in (
        lambda: search(home, SearchKeywords("robot")),
        lambda: vsearch(home, SearchVectors("robot"), Hashed()),
        lambda: query(home, SearchHybrid("robot"), Hashed()),
        lambda: unresolved(home, ListUnresolved()),
    ):
        with pytest.raises(InvalidRequestError, match="no collection is active") as refused:
            call()
        assert refused.value.remedy == "activate a collection by its name"


def test_an_inactive_collection_is_left_out_of_index_and_embed_unless_named(home: Path, tmp_path: Path) -> None:
    deactivate(home, DeactivateCollections(("diary",)))
    write(tmp_path / "diary" / "wednesday.md", "# Wednesday\n\nThe robot works.\n")
    assert [each.name for each in index(home, IndexCollections(), quiet).collections] == ["notes"]
    assert [each.name for each in embed(home, EmbedCollections(), quiet, Hashed()).collections] == ["notes"]
    assert [each.name for each in index(home, IndexCollections(("diary",)), quiet).collections] == ["diary"]
    activate(home, ActivateCollections(("diary",)))
    assert "diary/wednesday.md" in found(home, "works")
