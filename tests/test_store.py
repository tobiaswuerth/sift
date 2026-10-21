"""The SQLite index: connecting, the layout's version, collections, deleting documents, the vectors' version."""

import sqlite3
import time

import pytest

from sift import store
from sift.search import Index


def doc_ids(db):
    return dict(db.execute("SELECT path, id FROM docs"))


# --- connect and collections -------------------------------------------------


def test_connect_uses_home(home):
    store.connect()
    assert (home / "sift.db").exists()


def test_a_new_index_records_its_layout_version(home):
    assert store.connect(home).execute("PRAGMA user_version").fetchone() == (store.SCHEMA_VERSION,)


def test_an_index_from_an_older_sift_asks_to_be_rebuilt(home):
    home.mkdir()
    old = sqlite3.connect(home / "sift.db")
    old.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, path TEXT)")  # sift 0.2 had no layout version
    old.commit()
    old.close()
    with pytest.raises(SystemExit, match=r"older sift.*sift\.db.*sift add"):
        store.connect(home)


def test_an_index_from_a_newer_sift_asks_for_an_upgrade(home):
    home.mkdir()
    newer = sqlite3.connect(home / "sift.db")
    newer.execute(f"PRAGMA user_version = {store.SCHEMA_VERSION + 1}")
    newer.close()
    with pytest.raises(SystemExit, match="newer sift"):
        store.connect(home)


# The keyword index of sift 0.7 and older: title and text, no front matter.
KEYWORD_INDEX_0_7 = """DROP TABLE fts; DROP VIEW chunk_text;
CREATE VIEW chunk_text AS SELECT c.id AS id, d.title AS title, substr(d.text, c.pos + 1, c.len) AS text
    FROM chunks c JOIN docs d ON d.id = c.doc;
CREATE VIRTUAL TABLE fts USING fts5(title, text, content='chunk_text', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2');
INSERT INTO fts(fts) VALUES('rebuild');
PRAGMA user_version = 2;"""


def test_an_index_of_sift_0_7_gets_front_matter_into_its_keyword_index(embedded, home, capsys):
    # Only the keyword index is rebuilt, in minutes: embedding 3 million abstracts again would take hours.
    db, _ = embedded
    vectors = db.execute("SELECT count(*) FROM vectors").fetchone()
    db.executescript(KEYWORD_INDEX_0_7)
    db.close()
    db = store.connect(home)
    assert db.execute("PRAGMA user_version").fetchone() == (store.SCHEMA_VERSION,)
    assert db.execute("SELECT count(*) FROM vectors").fetchone() == vectors
    assert [h["id"] for h in Index(db).search("cs.GR")] == ["notes/robots/brush.md"]
    assert "updating the keyword index" in capsys.readouterr().err


def test_add_collection_defaults_name_to_folder(tmp_path, notes):
    db = store.connect(tmp_path / "h")
    assert store.add_collection(db, notes) == "notes"


@pytest.mark.parametrize("name", ["a/b", "a b", "../x", ""])
def test_add_collection_rejects_bad_names(tmp_path, notes, name):
    db = store.connect(tmp_path / "h")
    with pytest.raises(SystemExit):
        store.add_collection(db, notes, name or "/")


def test_add_collection_errors(corpus, tmp_path):
    db = corpus[0]
    (tmp_path / "other").mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        store.add_collection(db, tmp_path / "other", "notes")
    with pytest.raises(SystemExit, match="not a directory"):
        store.add_collection(db, tmp_path / "missing", "x")


def test_a_folder_is_one_collection(corpus):
    # Registered twice, every document of it was found twice.
    db, notes = corpus
    with pytest.raises(SystemExit, match=f"{notes.name} is the collection 'notes' already"):
        store.add_collection(db, notes / ".." / notes.name, "again")


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
