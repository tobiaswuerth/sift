"""The SQLite index: connecting, the layout's version, collections, deleting documents, the vectors' version."""

import time

import pytest

from sift import store


def doc_ids(db):
    return dict(db.execute("SELECT path, id FROM docs"))


# --- connect and collections -------------------------------------------------


def test_connect_uses_home(home):
    store.connect()
    assert (home / "sift.db").exists()


def test_add_collection_defaults_name_to_folder(tmp_path, notes):
    db = store.connect(tmp_path / "h")
    assert store.add_collection(db, notes) == "notes"


@pytest.mark.parametrize("name", ["a/b", "a b", "../x", ""])
def test_add_collection_rejects_bad_names(tmp_path, notes, name):
    db = store.connect(tmp_path / "h")
    with pytest.raises(SystemExit):
        store.add_collection(db, notes, name or "/")


def test_add_collection_errors(corpus, tmp_path):
    db, notes = corpus
    with pytest.raises(SystemExit, match="already exists"):
        store.add_collection(db, notes, "notes")
    with pytest.raises(SystemExit, match="not a directory"):
        store.add_collection(db, tmp_path / "missing", "x")


def test_remove_collection(corpus):
    db, root = corpus
    assert store.remove_collection(db, "notes") == 3  # documents removed
    assert (root / "rl.md").exists()  # only the index changes, never the files
    for table in ("collections", "sources", "docs", "chunks", "fts", "vectors"):
        assert db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    with pytest.raises(SystemExit, match="no collection"):
        store.remove_collection(db, "notes")


def test_delete_marks_vectors_changed(corpus):
    db, _ = corpus
    before = store.get_meta(db, "version")
    store.delete_docs(db, [])
    assert store.get_meta(db, "version") == before
    store.delete_docs(db, list(doc_ids(db).values())[:1])
    assert store.get_meta(db, "version") != before


def test_touch_changes_version_within_one_clock_tick(corpus, monkeypatch):
    db, _ = corpus
    monkeypatch.setattr(time, "time_ns", lambda: 1)  # before Python 3.13 the Windows clock ticks every 15.6 ms
    store.touch(db)
    before = store.get_meta(db, "version")
    store.touch(db)
    assert store.get_meta(db, "version") != before


def test_add_collection_expands_the_home_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # POSIX
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    (tmp_path / "notes").mkdir()
    db = store.connect(tmp_path / "h")
    store.add_collection(db, "~/notes")
    assert db.execute("SELECT path FROM collections").fetchone() == (str((tmp_path / "notes").resolve()),)


def test_add_collection_without_a_folder_name_asks_for_one(tmp_path):
    db = store.connect(tmp_path / "h")
    with pytest.raises(SystemExit, match="--name"):
        store.add_collection(db, tmp_path.anchor)  # a drive or file system root has no name
