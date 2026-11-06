"""The catalog: its layout, adding and removing collections, where their stores lie and moving them, and which are
active."""

import os
import sqlite3
from pathlib import Path

import pytest

from sift import catalog as catalog_module
from sift import console, indexing, store
from sift.catalog import Catalog
from sift.search import Index

from .conftest import write


def ids(hits):
    return [h["id"] for h in hits]


# --- the catalog ---------------------------------------------------------------


def test_the_catalog_lies_in_the_home_folder(home):
    Catalog().close()
    assert (home / "sift.db").exists()


def test_a_new_catalog_records_its_layout_version(home):
    assert Catalog(home).db.execute("PRAGMA user_version").fetchone() == (catalog_module.SCHEMA_VERSION,)


def test_an_index_from_an_older_sift_asks_to_be_rebuilt(home):
    home.mkdir()
    old = sqlite3.connect(home / "sift.db")
    old.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, path TEXT)")  # sift 0.2 had no layout version
    old.commit()
    old.close()
    with pytest.raises(SystemExit, match=r"older sift.*sift\.db.*sift add"):
        Catalog(home)


def test_a_catalog_from_a_newer_sift_asks_for_an_upgrade(home):
    home.mkdir()
    newer = sqlite3.connect(home / "sift.db")
    newer.execute(f"PRAGMA user_version = {catalog_module.SCHEMA_VERSION + 1}")
    newer.close()
    with pytest.raises(SystemExit, match="newer sift"):
        Catalog(home)


# --- adding collections --------------------------------------------------------


def test_add_defaults_the_name_to_the_folders(home, notes):
    assert Catalog(home).add(notes) == "notes"


def test_a_collections_store_lies_in_the_home_folder(home, notes):
    catalog = Catalog(home)
    catalog.add(notes)
    assert catalog.get("notes").store == str(home / "collections" / "notes.sqlite")
    assert (home / "collections" / "notes.sqlite").is_file()


def test_a_store_goes_into_the_folder_given(home, notes, tmp_path):
    # A large index goes on another drive.
    catalog = Catalog(home)
    catalog.add(notes, into=tmp_path / "big")
    assert catalog.get("notes").store == str(tmp_path / "big" / "notes.sqlite")
    indexing.index(catalog)
    assert ids(Index(catalog).search("reward")) == ["notes/rl.md"]


def test_a_store_takes_the_file_name_given(home, notes, tmp_path):
    catalog = Catalog(home)
    catalog.add(notes, "notes", into=tmp_path / "data_notes.sqlite")
    assert catalog.get("notes").store == str(tmp_path / "data_notes.sqlite")


def test_a_store_never_takes_a_file_there_already(home, notes, tmp_path):
    (tmp_path / "x.sqlite").write_bytes(b"something")
    catalog = Catalog(home)
    with pytest.raises(SystemExit, match="exists already"):
        catalog.add(notes, into=tmp_path / "x.sqlite")
    assert catalog.collections() == []
    assert (tmp_path / "x.sqlite").read_bytes() == b"something"


def test_a_store_of_a_collection_on_a_missing_drive_is_not_given_away(home, notes, tmp_path):
    catalog = Catalog(home)
    catalog.add(notes, into=tmp_path / "x.sqlite")
    (tmp_path / "x.sqlite").unlink()  # its drive isn't connected
    write(tmp_path / "other" / "a.md", "# A\n")
    with pytest.raises(SystemExit, match="exists already"):
        catalog.add(tmp_path / "other", into=tmp_path / "x.sqlite")


def test_names_differing_in_case_only_never_share_a_store(home, tmp_path):
    # On Windows and macOS, Notes.sqlite and notes.sqlite are one file.
    write(tmp_path / "upper" / "a.md", "# A\n")
    write(tmp_path / "lower" / "a.md", "# a\n")
    catalog = Catalog(home)
    catalog.add(tmp_path / "upper", "Notes")
    if (home / "collections" / "notes.sqlite").exists():  # a file system that ignores case
        with pytest.raises(SystemExit, match="exists already"):
            catalog.add(tmp_path / "lower", "notes")
    else:
        catalog.add(tmp_path / "lower", "notes")
        assert len({c.store for c in catalog.collections()}) == 2


@pytest.mark.parametrize("name", ["a/b", "a b", "../x", ""])
def test_add_rejects_bad_names(home, notes, name):
    with pytest.raises(SystemExit):
        Catalog(home).add(notes, name or "/")


def test_a_collection_can_take_titles_from_file_names(home, notes):
    catalog = Catalog(home)
    catalog.add(notes, fields={"title_from": "name"})
    assert catalog.get("notes").choices == {"title_from": "name"}


@pytest.mark.parametrize("choice", ["file", "", None, 1])
def test_titles_come_from_headings_or_file_names(home, notes, choice):
    with pytest.raises(SystemExit, match="titles come from 'heading' or 'name'"):
        Catalog(home).add(notes, fields={"title_from": choice})


def test_the_collection_at_a_folder(corpus, tmp_path):
    catalog, notes = corpus
    assert catalog.collection_at(notes / ".." / notes.name) == "notes"
    assert catalog.collection_at(tmp_path) is None


def test_add_errors(corpus, tmp_path):
    catalog = corpus[0]
    (tmp_path / "other").mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        catalog.add(tmp_path / "other", "notes")
    with pytest.raises(SystemExit, match="not a directory"):
        catalog.add(tmp_path / "missing", "x")
    assert [c.name for c in catalog.collections()] == ["notes"]


def test_a_folder_is_one_collection(corpus):
    # Registered twice, every document of it was found twice.
    catalog, notes = corpus
    with pytest.raises(SystemExit, match=f"{notes.name} is the collection 'notes' already"):
        catalog.add(notes / ".." / notes.name, "again")


def test_a_folder_inside_a_collection_or_around_one_is_refused(corpus):
    # Its files would be indexed twice, once in each collection, and found twice.
    catalog, notes = corpus
    with pytest.raises(SystemExit, match=r"robots is inside the collection 'notes' \(.*\): its files would be "):
        catalog.add(notes / "robots", "robots")
    with pytest.raises(SystemExit, match=r"holds the collection 'notes' \(.*\): .* --nested"):
        catalog.add(notes.parent, "everything")
    assert catalog.add(notes / "robots", "robots", nested=True) == "robots"


def test_add_expands_the_home_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # POSIX
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    (tmp_path / "notes").mkdir()
    catalog = Catalog(tmp_path / "h")
    catalog.add("~/notes")
    assert catalog.get("notes").path == str((tmp_path / "notes").resolve())


def test_add_without_a_folder_name_asks_for_one(tmp_path):
    with pytest.raises(SystemExit, match="--name"):
        Catalog(tmp_path / "h").add(tmp_path.anchor)  # a drive or file system root has no name


def test_an_unknown_collection_names_those_there_are(corpus):
    with pytest.raises(LookupError, match="no collection named 'nope'; the collections are notes"):
        corpus[0].get("nope")
    with pytest.raises(LookupError, match="none is registered yet"):
        Catalog(corpus[0].home / "other").get("nope")


# --- removing them -------------------------------------------------------------


def test_remove_deletes_the_store_and_keeps_the_files(corpus):
    catalog, root = corpus
    stored = catalog.get("notes").store
    assert catalog.remove("notes") == 3  # documents its store held
    assert (root / "rl.md").exists()  # only the index changes, never the files
    assert catalog.collections() == []
    assert not os.path.exists(stored)
    with pytest.raises(LookupError, match="no collection"):
        catalog.remove("notes")


def test_a_collection_whose_store_is_missing_can_be_removed(corpus, capsys):
    catalog, _ = corpus
    stored = catalog.get("notes").store
    catalog.release(catalog.get("notes").id)
    os.remove(stored)
    assert catalog.remove("notes") is None
    assert "its index is missing" in capsys.readouterr().err
    assert catalog.collections() == []


def test_an_id_is_never_given_twice(corpus, tmp_path):
    # Ids of documents and chunks start at their collection's: a new collection under a removed one's id could meet
    # codes a running server still holds of it.
    catalog, root = corpus
    removed = catalog.get("notes").id
    catalog.remove("notes")
    catalog.add(root, "again")
    assert catalog.get("again").id > removed


# --- active or not -------------------------------------------------------------


def test_collections_are_active_until_deactivated(corpus, tmp_path):
    catalog, _ = corpus
    write(tmp_path / "other" / "a.md", "# Other\n")
    catalog.add(tmp_path / "other", "other")
    assert [c.active for c in catalog.collections()] == [True, True]
    assert catalog.set_active(["other", "notes", "other"], False) == ["other", "notes"]
    assert [c.active for c in catalog.collections()] == [False, False]
    assert catalog.set_active(["notes"], False) == []  # inactive already
    assert catalog.set_active(["notes"], True) == ["notes"]
    assert {c.name: c.active for c in catalog.collections()} == {"notes": True, "other": False}


def test_an_unknown_name_changes_nothing(corpus):
    catalog, _ = corpus
    with pytest.raises(LookupError, match="'nope'"):
        catalog.set_active(["notes", "nope"], False)
    assert catalog.get("notes").active


def test_a_collection_deactivated_lets_go_of_its_store(corpus):
    # So that it can be moved, and holds no memory in a server.
    catalog, _ = corpus
    catalog.store("notes")
    catalog.set_active(["notes"], False)
    with store.alone(catalog.get("notes").store, "notes"):
        pass


# --- moving stores -------------------------------------------------------------


@pytest.fixture
def moved(corpus, fake_models):
    """The notes, embedded."""
    from sift import embed

    embed.embed_pending(corpus[0])
    return corpus


def searches(catalog):
    index = Index(catalog)
    return ids(index.search("reward")), ids(index.vsearch("reward shaping", k=1))


def test_a_store_moves_to_another_folder(moved, tmp_path):
    catalog, _ = moved
    old, before = catalog.get("notes").store, searches(catalog)
    assert catalog.move([], tmp_path / "elsewhere") == ["notes"]  # all of them
    assert catalog.get("notes").store == str(tmp_path / "elsewhere" / "notes.sqlite")
    assert not os.path.exists(old)
    assert searches(catalog) == before == (["notes/rl.md"], ["notes/rl.md"])


def test_a_store_moves_to_a_file_of_another_name(moved, tmp_path):
    catalog, _ = moved
    catalog.move(["notes"], tmp_path / "data_notes.sqlite")
    assert catalog.get("notes").store == str(tmp_path / "data_notes.sqlite")
    assert searches(catalog) == (["notes/rl.md"], ["notes/rl.md"])


def test_a_store_moves_to_another_drive_as_a_copy(moved, tmp_path, monkeypatch, capsys):
    # Another drive has no second name for a file: it is copied, through SQLite, page by page.
    catalog, _ = moved
    before = searches(catalog)
    monkeypatch.setattr(os, "link", lambda *args: (_ for _ in ()).throw(OSError(18, "Invalid cross-device link")))
    monkeypatch.setattr(catalog_module, "COPY_PAGES", 2)
    catalog.move(["notes"], tmp_path / "other drive")
    assert "notes: copying [####################] 100%" in capsys.readouterr().err
    assert not list((tmp_path / "other drive").glob("*.part"))
    assert searches(catalog) == before


def test_a_copy_interrupted_leaves_the_store_where_it_was(moved, tmp_path, monkeypatch):
    catalog, _ = moved
    old = catalog.get("notes").store
    monkeypatch.setattr(os, "link", lambda *args: (_ for _ in ()).throw(OSError(18, "Invalid cross-device link")))
    monkeypatch.setattr(catalog_module, "COPY_PAGES", 1)
    real = console.Progress.update

    def interrupted(progress, done):
        if done:  # a page copied
            raise KeyboardInterrupt
        real(progress, done)

    monkeypatch.setattr(console.Progress, "update", interrupted)
    with pytest.raises(KeyboardInterrupt):
        catalog.move(["notes"], tmp_path / "other drive")
    monkeypatch.setattr(console.Progress, "update", real)  # not undo(): that would also undo closing the databases
    assert catalog.get("notes").store == old
    assert not list((tmp_path / "other drive").iterdir())
    assert ids(Index(catalog).search("reward")) == ["notes/rl.md"]


def test_a_store_another_sift_has_open_stays(moved, tmp_path):
    catalog, _ = moved
    other = Catalog(catalog.home)
    other.store("notes")  # a server, say
    with pytest.raises(SystemExit, match="another sift has its index open"):
        catalog.move(["notes"], tmp_path / "elsewhere")
    assert catalog.get("notes").store == str(catalog.home / "collections" / "notes.sqlite")
    other.close()


def test_a_store_is_not_moved_onto_a_file(moved, tmp_path):
    catalog, _ = moved
    write(tmp_path / "elsewhere" / "notes.sqlite", "mine")
    with pytest.raises(SystemExit, match="exists already"):
        catalog.move(["notes"], tmp_path / "elsewhere")
    assert (tmp_path / "elsewhere" / "notes.sqlite").read_text() == "mine"


def test_several_stores_move_into_a_folder_only(moved, tmp_path):
    catalog, _ = moved
    write(tmp_path / "other" / "a.md", "# A\n")
    catalog.add(tmp_path / "other", "other")
    with pytest.raises(SystemExit, match="file of one collection's index"):
        catalog.move([], tmp_path / "all.sqlite")
    assert sorted(catalog.move([], tmp_path / "all")) == ["notes", "other"]


def test_a_store_where_it_is_already_stays(moved, capsys):
    catalog, _ = moved
    assert catalog.move(["notes"], catalog.home / "collections") == []
    assert "its index is" in capsys.readouterr().err


def test_a_missing_store_cannot_be_moved(moved, tmp_path):
    catalog, _ = moved
    catalog.release(catalog.get("notes").id)
    os.remove(catalog.get("notes").store)
    with pytest.raises(LookupError, match="its index is missing"):
        catalog.move(["notes"], tmp_path / "elsewhere")


def test_a_store_moved_by_another_sift_is_opened_where_it_is_now(moved, tmp_path):
    # A server keeps running while `sift move` moves a store it doesn't have open.
    catalog, _ = moved
    server = Index(Catalog(catalog.home))
    server.catalog.release(server.catalog.get("notes").id)
    catalog.move(["notes"], tmp_path / "elsewhere")
    assert ids(server.search("reward")) == ["notes/rl.md"]
    assert server.catalog._stores[server.catalog.get("notes").id][0] == str(tmp_path / "elsewhere" / "notes.sqlite")


def test_a_store_that_could_not_be_made_leaves_nothing(home, notes, monkeypatch):
    catalog = Catalog(home)
    monkeypatch.setattr(store, "make", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        catalog.add(notes)
    assert catalog.collections() == []
    assert not (home / "collections" / "notes.sqlite").exists()


def refusing_to_delete(monkeypatch, name):
    """Files named `name` can't be deleted, as Windows refuses while another program has them open."""
    real = Path.unlink

    def unlink(path, missing_ok=False):
        if path.name == name:
            raise PermissionError(13, "The process cannot access the file")
        return real(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)


def test_a_store_still_open_elsewhere_is_left_to_delete(corpus, monkeypatch, capsys):
    catalog, _ = corpus
    refusing_to_delete(monkeypatch, "notes.sqlite")
    catalog.remove("notes")
    assert "notes.sqlite could not be deleted, as another sift has it open" in capsys.readouterr().err
    assert catalog.collections() == []


def test_a_store_moved_but_still_open_at_its_old_place_is_left_to_delete(moved, tmp_path, monkeypatch, capsys):
    catalog, _ = moved
    refusing_to_delete(monkeypatch, "notes.sqlite")
    catalog.move(["notes"], tmp_path / "data_notes.sqlite")
    assert "notes: moved, but" in capsys.readouterr().err
    assert catalog.get("notes").store == str(tmp_path / "data_notes.sqlite")
