"""A collection's store: its layout, its ids, the vectors' codes, the counts of its rows, deleting documents, the
vectors' version, and holding it alone."""

import sqlite3
import time

import numpy as np
import pytest

from sift import embed, indexing, store, vectors
from sift.catalog import Catalog

from .conftest import write


def doc_ids(db):
    return dict(db.execute("SELECT path, id FROM docs"))


# --- layout --------------------------------------------------------------------


def test_a_new_store_records_its_layout_and_collection(tmp_path):
    db = store.connect(tmp_path / "a.sqlite", 7, create=True)
    assert db.execute("PRAGMA user_version").fetchone() == (store.SCHEMA_VERSION,)
    assert store.get_meta(db, "collection") == "7"
    assert all(store.get_meta(db, key) == "1" for key in store.READ_AGAIN)  # nothing an older sift skipped


def test_a_collections_ids_start_at_its_own(corpus):
    # Unique across collections, an id says whose it is: results of several stores merge as they are.
    catalog, _ = corpus
    (cid,) = [c.id for c in catalog.collections()]
    db = catalog.store("notes")
    for table in ("docs", "chunks"):
        ids = [i for (i,) in db.execute(f"SELECT id FROM {table}")]
        assert ids
        assert all(store.owner(i) == cid for i in ids)
        assert min(ids) == store.first_id(cid) + 1


def test_ids_are_never_handed_out_twice(corpus):
    # SQLite gave the ids of the newest deleted documents to the next ones, and a server kept the old vectors under
    # them: ids grow now, whatever is deleted.
    catalog, root = corpus
    db = catalog.store("notes")
    newest = max(doc_ids(db).values())
    store.delete_docs(db, [newest])
    db.commit()
    write(root / "new.md", "# New\n")
    indexing.index(catalog)
    assert doc_ids(db)["new.md"] > newest


def test_a_missing_store_is_not_made_anew(tmp_path):
    # A store on a drive that isn't connected: made anew, empty, it would seem every document deleted.
    with pytest.raises(FileNotFoundError, match="not found"):
        store.connect(tmp_path / "gone.sqlite", 1)
    assert not (tmp_path / "gone.sqlite").exists()


def test_a_store_of_another_collection_is_refused(tmp_path):
    store.connect(tmp_path / "a.sqlite", 1, create=True).close()
    with pytest.raises(SystemExit, match=r"index of another collection \(id 1, not 2\)"):
        store.connect(tmp_path / "a.sqlite", 2)


@pytest.mark.parametrize("version", [0, 5])
def test_a_file_that_is_no_store_is_refused(tmp_path, version):
    other = sqlite3.connect(tmp_path / "x.sqlite")
    other.execute("CREATE TABLE t(x)")
    other.execute(f"PRAGMA user_version = {version}")
    other.close()
    with pytest.raises(SystemExit, match="is not the index of a collection"):
        store.connect(tmp_path / "x.sqlite", 1)


def test_a_store_of_a_newer_sift_asks_for_an_upgrade(tmp_path):
    newer = sqlite3.connect(tmp_path / "a.sqlite")
    newer.execute(f"PRAGMA user_version = {store.SCHEMA_VERSION + 1}")
    newer.close()
    with pytest.raises(SystemExit, match="newer sift"):
        store.connect(tmp_path / "a.sqlite", 1)


# --- codes ---------------------------------------------------------------------


def codes_and_vectors(db):
    """The codes a store holds, and the codes of the vectors it holds: the same when they agree."""
    held = dict(db.execute("SELECT chunk, bits FROM codes"))
    rows = db.execute("SELECT chunk, vec FROM vectors ORDER BY chunk").fetchall()
    dim = int(store.get_meta(db, "dim"))
    expected = {
        chunk: vectors.codes(np.frombuffer(vec, np.float16).reshape(1, dim))[0].tobytes() for chunk, vec in rows
    }
    return held, expected


def test_vectors_are_stored_with_their_codes(embedded):
    held, expected = codes_and_vectors(embedded[0].store("notes"))
    assert held == expected
    assert len(held) == 3


def test_codes_are_added_to_vectors_without_them(embedded, capsys):
    # As an index of sift 1.6 is split (see upgrade): from the vectors held, nothing embedded again.
    db = embedded[0].store("notes")
    db.execute("DELETE FROM codes")
    db.commit()
    capsys.readouterr()
    store.add_codes(db)
    held, expected = codes_and_vectors(db)
    assert held == expected
    assert "adding the codes of 3 vectors" in capsys.readouterr().err
    store.add_codes(db)
    assert "codes" not in capsys.readouterr().err  # none left to add


def test_adding_codes_interrupted_in_a_batch_keeps_the_batches_before(embedded, monkeypatch):
    db = embedded[0].store("notes")
    db.execute("DELETE FROM codes")
    db.commit()
    monkeypatch.setattr(store, "CODES_BATCH", 1)
    calls, real = [], store._put_codes

    def interrupted(*args):
        calls.append(1)
        real(*args)
        if len(calls) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(store, "_put_codes", interrupted)
    with pytest.raises(KeyboardInterrupt):
        store.add_codes(db)
    monkeypatch.setattr(store, "_put_codes", real)  # not undo(): that would also undo closing the databases opened
    assert db.execute("SELECT count(*) FROM codes").fetchone() == (1,)
    store.add_codes(db)
    held, expected = codes_and_vectors(db)
    assert held == expected


def test_codes_are_of_the_vectors_as_they_are_when_written(embedded):
    # Another sift may re-embed a chunk between a batch being read and its codes written: each code must be that of
    # the vector the store holds afterwards.
    catalog, _ = embedded
    db = catalog.store("notes")
    db.execute("DELETE FROM codes")
    db.commit()
    chunk, vec = db.execute("SELECT chunk, vec FROM vectors LIMIT 1").fetchone()
    flipped, path = (-np.frombuffer(vec, np.float16)).tobytes(), catalog.get("notes").store

    class Racing:
        def __getattr__(self, name):
            return getattr(db, name)

        def execute(self, sql, *args):
            if sql == "BEGIN IMMEDIATE":  # the other sift writes just before this takes the lock
                other = sqlite3.connect(path)
                other.execute("UPDATE vectors SET vec = ? WHERE chunk = ?", (flipped, chunk))
                other.commit()
                other.close()
            return db.execute(sql, *args)

    store.add_codes(Racing())
    held, expected = codes_and_vectors(db)
    assert held == expected


def test_vectors_of_another_length_get_no_code(embedded):
    # A vector of another length than the store's can't be searched with the others.
    db = embedded[0].store("notes")
    db.execute("DELETE FROM codes")
    odd = db.execute("SELECT max(chunk) FROM vectors").fetchone()[0]
    db.execute("UPDATE vectors SET vec = ? WHERE chunk = ?", (bytes(10), odd))
    db.commit()
    store.add_codes(db)
    assert [chunk for (chunk,) in db.execute("SELECT chunk FROM codes")] == sorted(
        chunk for (chunk,) in db.execute("SELECT chunk FROM vectors WHERE chunk != ?", (odd,))
    )


def test_codes_go_with_their_vectors(embedded):
    db = embedded[0].store("notes")
    store.delete_docs(db, [doc_ids(db)["rl.md"]])
    held, expected = codes_and_vectors(db)
    assert held == expected
    assert len(held) == 2


def test_codes_go_with_a_reset(embedded, fake_models, monkeypatch):
    catalog, _ = embedded

    def interrupted(texts, query=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(fake_models, "encode", interrupted)
    with pytest.raises(KeyboardInterrupt):
        embed.embed_pending(catalog, reset=True)
    assert catalog.store("notes").execute("SELECT count(*) FROM codes").fetchone() == (0,)


# --- counts --------------------------------------------------------------------


def held(db):
    """How many rows of each counted table a store holds, counted."""
    return {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in store.COUNTED}


def as_sift_1_8_left_it(db):
    """A store as sift 1.8 made it: layout 6, its rows not counted."""
    triggers = db.execute("SELECT name FROM sqlite_master WHERE type = 'trigger' AND name != 'codes_go_with_vectors'")
    for (name,) in triggers.fetchall():
        db.execute(f"DROP TRIGGER {name}")
    db.execute("DROP TABLE counts")
    db.execute("PRAGMA user_version = 6")
    db.commit()


def test_a_new_store_counts_nothing(tmp_path):
    db = store.connect(tmp_path / "a.sqlite", 1, create=True)
    assert store.counts(db) == {"docs": 0, "chunks": 0, "vectors": 0}


def add_file(catalog, root):
    write(root / "new.md", "# New\n\nA new note about pigments.\n")
    indexing.index(catalog)


def change_file(catalog, root):
    write(root / "rl.md", "# Reward shaping\n\nReward shaping, written again.\n\n## More\n\nAnother passage.\n")
    indexing.index(catalog)


def remove_file(catalog, root):
    (root / "rl.md").unlink()
    indexing.index(catalog)


def delete_document(catalog, root):
    db = catalog.store("notes")
    store.delete_docs(db, [doc_ids(db)["rl.md"]])
    db.commit()


def write_vector_again(catalog, root):
    db = catalog.store("notes")
    chunk, vec = db.execute("SELECT chunk, vec FROM vectors LIMIT 1").fetchone()
    store.put_vectors(db, [chunk], -np.frombuffer(vec, np.float16).reshape(1, -1))
    db.commit()


def reset(catalog, root):
    embed.embed_pending(catalog, reset=True)


def drop_broken_vector(catalog, root):
    db = catalog.store("notes")
    chunk, vec = db.execute("SELECT chunk, vec FROM vectors LIMIT 1").fetchone()
    db.execute(
        "UPDATE vectors SET vec = ? WHERE chunk = ?", (np.full(len(vec) // 2, np.nan, np.float16).tobytes(), chunk)
    )
    db.execute("DELETE FROM meta WHERE key = 'vectors_checked'")
    db.commit()
    embed.embed_pending(catalog)


@pytest.mark.parametrize(
    "change",
    [add_file, change_file, remove_file, delete_document, write_vector_again, reset, drop_broken_vector],
)
def test_a_store_keeps_its_counts_through_every_change(embedded, change):
    # So that a status needs no pass over a table: SQLite keeps no counts, and counting a large store's vectors read
    # tens of gigabytes.
    catalog, root = embedded
    change(catalog, root)
    db = catalog.store("notes")
    assert store.counts(db) == held(db)
    assert held(db)["docs"] > 0


def test_a_vector_written_again_replaces_the_old_one(embedded):
    db = embedded[0].store("notes")
    chunk, vec = db.execute("SELECT chunk, vec FROM vectors LIMIT 1").fetchone()
    flipped = -np.frombuffer(vec, np.float16).reshape(1, -1)
    store.put_vectors(db, [chunk], flipped)
    db.commit()
    assert db.execute("SELECT vec FROM vectors WHERE chunk = ?", (chunk,)).fetchone() == (flipped.tobytes(),)
    stored, expected = codes_and_vectors(db)
    assert stored == expected
    assert store.counts(db)["vectors"] == 3


def test_counts_go_back_with_a_change_undone(embedded):
    # Counted in the transaction of the change itself: whatever stops a change, rows and counts stay as one.
    db = embedded[0].store("notes")
    before = store.counts(db)
    store.delete_docs(db, list(doc_ids(db).values()))
    db.rollback()
    assert store.counts(db) == before == held(db)


def test_a_store_of_sift_1_8_counts_its_rows_once(embedded, capsys):
    catalog, _ = embedded
    as_sift_1_8_left_it(catalog.store("notes"))
    before = held(catalog.store("notes"))
    catalog.release(catalog.get("notes").id)
    capsys.readouterr()
    db = catalog.store("notes")
    assert db.execute("PRAGMA user_version").fetchone() == (store.SCHEMA_VERSION,)
    assert store.counts(db) == before
    assert "counting the rows of" in capsys.readouterr().err
    catalog.release(catalog.get("notes").id)
    catalog.store("notes")
    assert "counting" not in capsys.readouterr().err  # once


def test_a_store_counted_meanwhile_is_counted_once(embedded, capsys):
    # Two sifts that open a store of 1.8 at once both see it uncounted; the second must find the counts there once it
    # has the store, rather than fail or count again.
    db = embedded[0].store("notes")
    capsys.readouterr()
    store._count_rows(db, "notes.sqlite")  # what the second runs, after the first has finished
    assert store.counts(db) == held(db)
    assert "counting" not in capsys.readouterr().err


def test_counting_stopped_leaves_the_store_as_sift_1_8_left_it(embedded, monkeypatch):
    catalog, _ = embedded
    as_sift_1_8_left_it(catalog.store("notes"))
    catalog.release(catalog.get("notes").id)

    class StoppedAfterDocs:
        def __iter__(self):
            yield "docs"
            raise KeyboardInterrupt

    real = store.COUNTED
    monkeypatch.setattr(store, "COUNTED", StoppedAfterDocs())
    with pytest.raises(KeyboardInterrupt):
        catalog.store("notes")
    monkeypatch.setattr(store, "COUNTED", real)  # not undo(): that would also undo closing the databases opened
    raw = sqlite3.connect(catalog.get("notes").store)
    assert raw.execute("PRAGMA user_version").fetchone() == (6,)
    assert raw.execute("SELECT count(*) FROM sqlite_master WHERE name = 'counts'").fetchone() == (0,)
    raw.close()
    assert store.counts(catalog.store("notes")) == held(catalog.store("notes"))  # counted on the next start


# --- version -------------------------------------------------------------------


def test_delete_marks_vectors_changed(corpus):
    db = corpus[0].store("notes")
    before = store.get_meta(db, "version")
    store.delete_docs(db, [])
    assert store.get_meta(db, "version") == before
    store.delete_docs(db, list(doc_ids(db).values())[:1])
    assert store.get_meta(db, "version") != before


def test_touch_changes_version_within_one_clock_tick(corpus, monkeypatch):
    db = corpus[0].store("notes")
    monkeypatch.setattr(time, "time_ns", lambda: 1)  # before Python 3.13 the Windows clock ticks every 15.6 ms
    store.touch(db)
    before = store.get_meta(db, "version")
    store.touch(db)
    assert store.get_meta(db, "version") != before


# --- holding a store alone -----------------------------------------------------


def test_a_store_held_alone_is_closed_to_every_other_sift(tmp_path):
    path = tmp_path / "a.sqlite"
    db = store.connect(path, 1, create=True)
    store.set_meta(db, "x", 1)
    db.commit()
    db.close()
    with store.alone(path, "a") as held:
        assert held.execute("SELECT value FROM meta WHERE key = 'x'").fetchone() == ("1",)
        log = tmp_path / "a.sqlite-wal"
        assert not log.exists() or log.stat().st_size == 0  # all of it in the file
        other = sqlite3.connect(path, timeout=0.1)
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            other.execute("SELECT * FROM meta").fetchall()
        other.close()
    assert store.connect(path, 1).execute("SELECT value FROM meta WHERE key = 'x'").fetchone() == ("1",)


def test_a_store_another_sift_has_open_is_not_held(home, notes):
    catalog = Catalog(home)
    catalog.add(notes, "notes")
    catalog.store("notes")  # a server, say, has it open
    with (
        pytest.raises(SystemExit, match="notes: another sift has its index open"),
        store.alone(catalog.get("notes").store, "notes"),
    ):
        pass
