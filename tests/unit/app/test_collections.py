import contextlib
from pathlib import Path

import pytest

from sift.api.collections import (
    AddCollection,
    CollectionAdded,
    CollectionStatus,
    Counted,
    Status,
    TitleFrom,
    Unavailable,
)
from sift.api.errors import InvalidRequestError
from sift.api.ids import CollectionId
from sift.app.collections import add, status
from sift.domain.text.notes import note
from sift.storage.database import writing
from sift.storage.documents import add_documents, put_source
from sift.storage.store import MISSING_REMEDY, open_store


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


def folder(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir(parents=True)
    return path


def test_a_folder_is_added_as_a_collection_named_after_it(tmp_path: Path, home: Path) -> None:
    notes = folder(tmp_path, "notes")
    assert add(home, AddCollection(str(notes))) == CollectionAdded(
        name="notes",
        folder=str(notes.resolve()),
        store=str((home / "collections" / "notes.sqlite").resolve()),
        title_from=TitleFrom.HEADING,
        vault=False,
    )


def test_the_name_place_and_titles_chosen_are_kept(tmp_path: Path, home: Path) -> None:
    request = AddCollection(
        str(folder(tmp_path, "notes")),
        name="mine",
        store=str(tmp_path / "stores"),
        title_from=TitleFrom.NAME,
    )
    added = add(home, request)
    assert (added.name, added.store, added.title_from) == (
        "mine",
        str((tmp_path / "stores" / "mine.sqlite").resolve()),
        TitleFrom.NAME,
    )


def test_an_obsidian_vault_is_titled_by_file_names_unless_told_otherwise(tmp_path: Path, home: Path) -> None:
    vault, other = folder(tmp_path, "vault"), folder(tmp_path, "other")
    for each in (vault, other):
        (each / ".obsidian").mkdir()
    assert add(home, AddCollection(str(vault))).title_from is TitleFrom.NAME
    chosen = add(home, AddCollection(str(other), title_from=TitleFrom.HEADING))
    assert (chosen.title_from, chosen.vault) == (TitleFrom.HEADING, True)


def test_a_folder_inside_a_collection_is_added_only_as_nested(tmp_path: Path, home: Path) -> None:
    notes = folder(tmp_path, "notes")
    inner = folder(notes, "inner")
    add(home, AddCollection(str(notes)))
    with pytest.raises(InvalidRequestError, match="lies inside the folder of the collection 'notes'"):
        add(home, AddCollection(str(inner)))
    assert add(home, AddCollection(str(inner), nested=True)).name == "inner"


def test_status_without_a_catalog_is_empty_and_makes_none(home: Path) -> None:
    assert status(home) == Status(str(home), ())
    assert not home.exists()


def test_status_counts_what_each_store_holds(tmp_path: Path, home: Path) -> None:
    papers = add(home, AddCollection(str(folder(tmp_path, "papers"))))
    add(home, AddCollection(str(folder(tmp_path, "Notes"))))
    store = Path(papers.store)
    with contextlib.closing(open_store(store, CollectionId(1))) as connection, writing(connection) as transaction:
        for name in ("a.md", "b.md"):
            add_documents(transaction, put_source(transaction, name, 1, 2), name, [note(b"# A\n\ntext\n", name)])
    (notes, counted) = status(home).collections
    assert notes == CollectionStatus(
        "Notes",
        str((tmp_path / "Notes").resolve()),
        str(home.resolve() / "collections" / "Notes.sqlite"),
        notes.contents,
    )
    assert isinstance(notes.contents, Counted)
    assert (notes.contents.documents, notes.contents.chunks) == (0, 0)
    assert counted.contents == Counted(documents=2, chunks=2, stored_bytes=store.stat().st_size)


def test_status_says_why_a_store_cannot_be_read(tmp_path: Path, home: Path) -> None:
    added = add(home, AddCollection(str(folder(tmp_path, "notes"))))
    Path(added.store).unlink()
    (only,) = status(home).collections
    assert only.contents == Unavailable(f"the store is missing: {added.store}", MISSING_REMEDY)
