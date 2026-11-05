"""The SQLite index: connecting, the layout's version, collections, deleting documents, the vectors' version."""

import sqlite3
import time

import numpy as np
import pytest

from sift import embed, store, vectors
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


def codes_and_vectors(db):
    """The codes the index holds, and the codes of the vectors it holds: the same when they agree."""
    held = dict(db.execute("SELECT chunk, bits FROM codes"))
    rows = db.execute("SELECT chunk, vec FROM vectors ORDER BY chunk").fetchall()
    dim = int(store.get_meta(db, "dim"))
    expected = {
        chunk: vectors.codes(np.frombuffer(vec, np.float16).reshape(1, dim))[0].tobytes() for chunk, vec in rows
    }
    return held, expected


def test_vectors_are_stored_with_their_codes(embedded):
    held, expected = codes_and_vectors(embedded[0])
    assert held == expected
    assert len(held) == 3


def test_an_index_of_sift_1_6_gets_its_codes_once(embedded, home, capsys):
    # Codes come from the vectors the index holds: no file is read, nothing embedded again.
    db, _ = embedded
    db.executescript("DROP TRIGGER codes_go_with_vectors; DROP TABLE codes; PRAGMA user_version = 4;")
    db.close()
    capsys.readouterr()
    db = store.connect(home)
    assert db.execute("PRAGMA user_version").fetchone() == (store.SCHEMA_VERSION,)
    held, expected = codes_and_vectors(db)
    assert held == expected
    assert "adding the codes of 3 vectors" in capsys.readouterr().err
    store.connect(home)
    assert "codes" not in capsys.readouterr().err  # once


def test_an_upgrade_stopped_half_way_goes_on(embedded, home):
    db, _ = embedded
    db.execute("DELETE FROM codes WHERE chunk = (SELECT max(chunk) FROM codes)")  # as if stopped before the last
    db.execute("PRAGMA user_version = 4")
    db.commit()
    db.close()
    held, expected = codes_and_vectors(store.connect(home))
    assert held == expected


def test_an_upgrade_interrupted_in_a_batch_keeps_the_batches_before(embedded, home, monkeypatch):
    db, _ = embedded
    db.executescript("DROP TRIGGER codes_go_with_vectors; DROP TABLE codes; PRAGMA user_version = 4;")
    db.close()
    monkeypatch.setattr(store, "CODES_BATCH", 1)
    calls, real = [], store._put_codes

    def interrupted(*args):
        calls.append(1)
        real(*args)
        if len(calls) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(store, "_put_codes", interrupted)
    with pytest.raises(KeyboardInterrupt):
        store.connect(home)
    monkeypatch.undo()
    db = sqlite3.connect(home / "sift.db")
    assert (db.execute("PRAGMA user_version").fetchone(), db.execute("SELECT count(*) FROM codes").fetchone()) == (
        (4,),
        (1,),
    )
    db.close()
    held, expected = codes_and_vectors(store.connect(home))
    assert held == expected


def test_an_upgrade_codes_the_vectors_as_they_are_when_it_writes(embedded, home, monkeypatch):
    # Another sift may re-embed a chunk between the upgrade reading a batch and writing its codes: each code must be
    # that of the vector the index holds afterwards.
    db, _ = embedded
    db.executescript("DROP TRIGGER codes_go_with_vectors; DROP TABLE codes; PRAGMA user_version = 4;")
    chunk, vec = db.execute("SELECT chunk, vec FROM vectors LIMIT 1").fetchone()
    db.close()
    flipped, real = (-np.frombuffer(vec, np.float16)).tobytes(), sqlite3.connect

    class Racing(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql == "BEGIN IMMEDIATE":  # the other sift writes just before the upgrade takes the lock
                other = real(home / "sift.db")
                other.execute("UPDATE vectors SET vec = ? WHERE chunk = ?", (flipped, chunk))
                other.commit()
                other.close()
            return super().execute(sql, *args)

    monkeypatch.setattr(store.sqlite3, "connect", lambda *args, **kwargs: real(*args, factory=Racing, **kwargs))
    held, expected = codes_and_vectors(store.connect(home))
    assert held == expected


def test_an_upgrade_skips_vectors_of_another_length(embedded, home):
    # A vector of another length than the index's can't be searched with the others: it gets no code, and the upgrade
    # still finishes.
    db, _ = embedded
    db.executescript("DROP TRIGGER codes_go_with_vectors; DROP TABLE codes; PRAGMA user_version = 4;")
    odd = db.execute("SELECT max(chunk) FROM vectors").fetchone()[0]
    db.execute("UPDATE vectors SET vec = ? WHERE chunk = ?", (bytes(10), odd))
    db.commit()
    db.close()
    db = store.connect(home)
    assert db.execute("PRAGMA user_version").fetchone() == (store.SCHEMA_VERSION,)
    assert [chunk for (chunk,) in db.execute("SELECT chunk FROM codes")] == sorted(
        chunk for (chunk,) in db.execute("SELECT chunk FROM vectors WHERE chunk != ?", (odd,))
    )


def test_codes_go_with_their_vectors(embedded):
    db, _ = embedded
    store.delete_docs(db, [doc_ids(db)["rl.md"]])
    held, expected = codes_and_vectors(db)
    assert held == expected
    assert len(held) == 2
    embed.embed_pending(db, reset=True)
    assert codes_and_vectors(db)[0] == codes_and_vectors(db)[1]


def test_add_collection_defaults_name_to_folder(tmp_path, notes):
    db = store.connect(tmp_path / "h")
    assert store.add_collection(db, notes) == "notes"


@pytest.mark.parametrize("name", ["a/b", "a b", "../x", ""])
def test_add_collection_rejects_bad_names(tmp_path, notes, name):
    db = store.connect(tmp_path / "h")
    with pytest.raises(SystemExit):
        store.add_collection(db, notes, name or "/")


def test_a_collection_can_take_titles_from_file_names(tmp_path, notes):
    db = store.connect(tmp_path / "h")
    store.add_collection(db, notes, fields={"title_from": "name"})
    assert db.execute("SELECT fields FROM collections").fetchone() == ('{"title_from": "name"}',)


@pytest.mark.parametrize("choice", ["file", "", None, 1])
def test_titles_come_from_headings_or_file_names(tmp_path, notes, choice):
    db = store.connect(tmp_path / "h")
    with pytest.raises(SystemExit, match="titles come from 'heading' or 'name'"):
        store.add_collection(db, notes, fields={"title_from": choice})


def test_the_collection_at_a_folder(corpus, tmp_path):
    db, notes = corpus
    assert store.collection_at(db, notes / ".." / notes.name) == "notes"
    assert store.collection_at(db, tmp_path) is None


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


def test_a_folder_inside_a_collection_or_around_one_is_refused(corpus):
    # Its files would be indexed twice, once in each collection, and found twice.
    db, notes = corpus
    with pytest.raises(SystemExit, match=r"robots is inside the collection 'notes' \(.*\): its files would be "):
        store.add_collection(db, notes / "robots", "robots")
    with pytest.raises(SystemExit, match=r"holds the collection 'notes' \(.*\): .* --nested"):
        store.add_collection(db, notes.parent, "everything")
    assert store.add_collection(db, notes / "robots", "robots", nested=True) == "robots"


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
