import contextlib
import sqlite3
import stat
import sys
from pathlib import Path

import pytest

from sift.api.collections import (
    ActivateCollections,
    AddCollection,
    CollectionAdded,
    CollectionStatus,
    Counted,
    ShowStatus,
    Status,
    TitleFrom,
    Unavailable,
)
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, NotFoundError
from sift.api.ids import CollectionId
from sift.api.indexing import IndexCollections
from sift.app.collections import activate, add, status
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.app.paths import ABSOLUTE_REMEDY
from sift.domain.text.notes import note
from sift.storage.catalog import forget_collection, known_collections
from sift.storage.database import PRIVATE_FILE, PRIVATE_FOLDER
from sift.storage.documents import add_documents, put_source
from sift.storage.keywords import writing_documents
from sift.storage.store import MISSING_REMEDY, open_store
from sift.storage.unusable import DAMAGED_REMEDY
from tests.damage import damage_table
from tests.files import undeletable, write
from tests.models import Hashed


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


def new(home: Path, request: AddCollection) -> CollectionAdded:
    added = add(home, request)
    assert isinstance(added, CollectionAdded)
    return added


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
        follow_links=False,
    )


def test_the_name_place_and_titles_chosen_are_kept(tmp_path: Path, home: Path) -> None:
    request = AddCollection(
        str(folder(tmp_path, "notes")),
        name="mine",
        store=str(tmp_path / "stores"),
        title_from=TitleFrom.NAME,
    )
    added = new(home, request)
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
    chosen = new(home, AddCollection(str(other), title_from=TitleFrom.HEADING))
    assert (chosen.title_from, chosen.vault) == (TitleFrom.HEADING, True)


def test_a_folder_inside_a_collection_is_added_only_as_nested(tmp_path: Path, home: Path) -> None:
    notes = folder(tmp_path, "notes")
    inner = folder(notes, "inner")
    add(home, AddCollection(str(notes)))
    with pytest.raises(InvalidRequestError, match="lies inside the folder of the collection 'notes'"):
        add(home, AddCollection(str(inner)))
    assert add(home, AddCollection(str(inner), nested=True)).name == "inner"


def test_a_store_its_removal_left_behind_is_made_anew_when_its_folder_is_added_again(
    tmp_path: Path, home: Path
) -> None:
    notes = folder(tmp_path, "notes")
    write(notes / "a.md", "# A\n\nAn apple.\n")
    added = new(home, AddCollection(str(notes)))
    index(home, IndexCollections(), lambda _: None)
    (one,) = known_collections(home)
    forget_collection(home, one.id)
    assert new(home, AddCollection(str(notes))).store == added.store
    (shown,) = status(home, ShowStatus()).collections
    assert isinstance(shown.contents, Counted)
    assert shown.contents.documents == 0


def test_a_store_its_removal_left_open_elsewhere_is_taken_by_an_add_once_closed(tmp_path: Path, home: Path) -> None:
    notes = folder(tmp_path, "notes")
    added = new(home, AddCollection(str(notes)))
    (one,) = known_collections(home)
    forget_collection(home, one.id)
    with undeletable(Path(added.store)), pytest.raises(InvalidRequestError, match="is open in another program"):
        add(home, AddCollection(str(notes)))
    assert new(home, AddCollection(str(notes))).store == added.store


def test_an_add_refused_for_a_database_in_its_place_leaves_nothing_beside_it(tmp_path: Path, home: Path) -> None:
    taken = home / "collections" / "notes.sqlite"
    taken.parent.mkdir(parents=True)
    with contextlib.closing(sqlite3.connect(taken)) as other:
        other.execute("PRAGMA journal_mode = WAL")
        other.execute("CREATE TABLE mine(x INTEGER)")
    with pytest.raises(InvalidRequestError, match="exists already"):
        add(home, AddCollection(str(folder(tmp_path, "notes"))))
    assert sorted(file.name for file in taken.parent.iterdir()) == ["notes.sqlite"]


def test_status_without_a_catalog_is_empty_and_makes_none(home: Path) -> None:
    assert status(home, ShowStatus()) == Status(str(home), ())
    assert not home.exists()


def test_status_counts_what_each_store_holds(tmp_path: Path, home: Path) -> None:
    papers = new(home, AddCollection(str(folder(tmp_path, "papers"))))
    add(home, AddCollection(str(folder(tmp_path, "Notes"))))
    store = Path(papers.store)
    with (
        contextlib.closing(open_store(store, CollectionId(1))) as connection,
        writing_documents(connection) as transaction,
    ):
        for name in ("a.md", "b.md"):
            add_documents(transaction, put_source(transaction, name, 1, 2), name, [note(b"# A\n\ntext\n", name)])
    (notes, counted) = status(home, ShowStatus()).collections
    assert notes == CollectionStatus(
        "Notes",
        str((tmp_path / "Notes").resolve()),
        str(home.resolve() / "collections" / "Notes.sqlite"),
        notes.contents,
    )
    assert isinstance(notes.contents, Counted)
    assert (notes.contents.documents, notes.contents.chunks) == (0, 0)
    assert counted.contents == Counted(2, 2, store.stat().st_size, unreadable=0, unreadable_files=())


def test_status_says_what_is_embedded_by_which_model_the_fields_held_and_what_is_searched(
    tmp_path: Path, home: Path
) -> None:
    root = tmp_path / "notes"
    write(root / "a.md", "---\nyear: 2024\nauthor: Ada\n---\n# A\n\ntext\n")
    write(root / "b.md", "---\ntags: [x]\n---\n# B\n\nmore\n")
    write(root / "c.md", "# C\n\nplain\n")
    add(home, AddCollection(str(root)))
    add(home, AddCollection(str(folder(tmp_path, "other"))))
    index(home, IndexCollections(), lambda _: None)
    embed(home, EmbedCollections(("notes",)), lambda _: None, Hashed())
    with contextlib.closing(sqlite3.connect(home / "catalog.sqlite")) as catalog, catalog:
        catalog.execute("UPDATE collections SET active = 0 WHERE name = 'other'")
    notes, other = status(home, ShowStatus()).collections
    assert isinstance(notes.contents, Counted)
    assert isinstance(other.contents, Counted)
    held = notes.contents
    assert (held.chunks, held.embedded, held.model, held.fields) == (3, 3, "hashed-words", ("author", "tags", "year"))
    assert (other.contents.embedded, other.contents.model, other.contents.fields) == (0, None, ())
    assert (notes.active, other.active) == (True, False)


def test_status_shows_the_collections_named(tmp_path: Path, home: Path) -> None:
    for name in ("a", "b"):
        add(home, AddCollection(str(folder(tmp_path, name))))
    assert [each.name for each in status(home, ShowStatus(("B", "b"))).collections] == ["b"]
    with pytest.raises(NotFoundError, match="no collection is named 'c'"):
        status(home, ShowStatus(("c",)))


def test_status_of_a_collection_named_where_there_is_no_catalog_makes_none(home: Path) -> None:
    with pytest.raises(NotFoundError, match="no collection is named 'c'"):
        status(home, ShowStatus(("c",)))
    assert not home.exists()


def test_status_of_a_store_damaged_where_it_is_read_says_so_and_shows_the_others(tmp_path: Path, home: Path) -> None:
    for name in ("a", "b"):
        add(home, AddCollection(str(folder(tmp_path, name))))
    damage_table(home / "collections" / "a.sqlite", "counts")
    damaged, sound = status(home, ShowStatus()).collections
    assert isinstance(damaged.contents, Unavailable)
    assert ("is damaged" in damaged.contents.message, damaged.contents.remedy) == (True, DAMAGED_REMEDY)
    assert isinstance(sound.contents, Counted)


def test_status_says_why_a_store_cannot_be_read(tmp_path: Path, home: Path) -> None:
    added = new(home, AddCollection(str(folder(tmp_path, "notes"))))
    Path(added.store).unlink()
    (only,) = status(home, ShowStatus()).collections
    assert only.contents == Unavailable(f"the store is missing: {added.store}", MISSING_REMEDY)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_the_home_its_stores_and_their_logs_are_the_user_s_alone(tmp_path: Path) -> None:
    home, folder = tmp_path / "home", tmp_path / "notes"
    folder.mkdir()
    (folder / "a.md").write_text("# A\n", encoding="utf-8")
    add(home, AddCollection(str(folder)))
    index(home, IndexCollections(), lambda _: None)
    for path in (home, home / "collections"):
        assert stat.S_IMODE(path.stat().st_mode) == PRIVATE_FOLDER, path
    files = [home / "catalog.sqlite", *sorted((home / "collections").glob("notes.sqlite*"))]
    assert all(stat.S_IMODE(path.stat().st_mode) == PRIVATE_FILE for path in files), files


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_a_home_and_stores_others_could_read_are_made_the_user_s_when_written(tmp_path: Path) -> None:
    home, folder = tmp_path / "home", tmp_path / "notes"
    folder.mkdir()
    (folder / "a.md").write_text("# A\n", encoding="utf-8")
    add(home, AddCollection(str(folder)))
    for path, shared in (
        (home, 0o755),
        (home / "catalog.sqlite", 0o644),
        (home / "collections" / "notes.sqlite", 0o644),
    ):
        path.chmod(shared)
    index(home, IndexCollections(), lambda _: None)
    activate(home, ActivateCollections(("notes",)))
    assert stat.S_IMODE(home.stat().st_mode) == PRIVATE_FOLDER
    assert {stat.S_IMODE(path.stat().st_mode) for path in home.rglob("*.sqlite")} == {PRIVATE_FILE}


@pytest.mark.parametrize(
    "folder, store, field",
    [("notes", None, "folder"), (None, "stores", "store")],
)
def test_a_folder_or_store_to_add_that_is_not_absolute_is_refused_by_its_field(
    tmp_path: Path, home: Path, folder: str | None, store: str | None, field: str
) -> None:
    asked = AddCollection(str(tmp_path) if folder is None else folder, store=store)
    with pytest.raises(InvalidRequestError, match="is no absolute path") as refused:
        add(home, asked)
    assert (refused.value.field, refused.value.remedy, known_collections(home)) == (field, ABSOLUTE_REMEDY, ())
