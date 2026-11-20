"""A collection's store: its layout, its ids, the vectors' codes, the counts of its rows, the last parts of its
documents' paths, deleting documents, the vectors' version, and holding it alone."""

import json
import random
import re
import sqlite3
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pytest

from sift import embed, indexing, postings, store, vectors
from sift.catalog import Catalog
from sift.search import Index

from .conftest import as_sift_1_12_left_it, keywords_agree, write


def doc_ids(db):
    return dict(db.execute("SELECT path, id FROM docs"))


# --- layout --------------------------------------------------------------------


def test_a_new_store_records_its_layout_and_collection(tmp_path):
    db = store.connect(tmp_path / "a.sqlite", 7, create=True)
    assert db.execute("PRAGMA user_version").fetchone() == (store.SCHEMA_VERSION,)
    assert store.get_meta(db, "collection") == "7"
    # nothing an older sift skipped or left out, to be read again
    assert all(store.get_meta(db, key) == "1" for key in (*store.READ_AGAIN, store.WAITED))


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


@pytest.mark.parametrize(
    "path, uri",
    [
        (PureWindowsPath(r"\\nas\share\sift\a.sqlite"), "file:////nas/share/sift/a.sqlite?mode=rw"),
        (PureWindowsPath(r"D:\sift\a b#.sqlite"), "file:///D:/sift/a%20b%23.sqlite?mode=rw"),
        (PurePosixPath("/home/me/a.sqlite"), "file:///home/me/a.sqlite?mode=rw"),
    ],
)
def test_a_store_is_opened_by_a_uri_sqlite_takes(path, uri):
    # A store on a network share, or on a drive mapped to one (Z:, which resolves to \\nas\share), was made and then
    # never opened again: SQLite refuses a URI that names a host, file://nas/share/..., as Path.as_uri writes it.
    assert store.uri(path, "rw") == uri


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

    def interrupted(batches, query=False):
        raise KeyboardInterrupt
        yield  # a generator, as encode_batches is

    monkeypatch.setattr(fake_models, "encode_batches", interrupted)
    with pytest.raises(KeyboardInterrupt):
        embed.embed_pending(catalog, reset=True)
    assert catalog.store("notes").execute("SELECT count(*) FROM codes").fetchone() == (0,)


# --- counts --------------------------------------------------------------------


def held(db):
    """How many rows of each counted table a store holds, counted."""
    return {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in store.COUNTED}


def as_sift_1_8_left_it(db):
    """A store as sift 1.8 made it: layout 6, its rows not counted."""
    as_sift_1_12_left_it(db)
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


# --- the keyword postings' tables and changes -------------------------------------

POSTINGS_TABLES = {"kw_segments", "kw_terms", "kw_blocks", "kw_lengths", "kw_masked", "kw_dirty", "kw_order"}


def tables(db):
    return {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def as_sift_1_9_left_it(db):
    """A store as sift 1.9 made it: layout 7, no postings of its own."""
    as_sift_1_12_left_it(db)
    for table in POSTINGS_TABLES:
        db.execute(f"DROP TABLE {table}")
    db.execute("DELETE FROM meta WHERE key IN ('kw_built', 'kw_change')")
    db.execute("PRAGMA user_version = 7")
    db.commit()


@pytest.fixture
def unfollowed(monkeypatch):
    """Postings that indexing leaves as they are, so that the marks it makes stay to be seen."""
    monkeypatch.setattr(postings, "update", lambda *args, **kwargs: None)


def test_a_new_store_has_the_postings_tables_unbuilt(tmp_path):
    db = store.connect(tmp_path / "a.sqlite", 1, create=True)
    assert tables(db) >= POSTINGS_TABLES
    assert store.get_meta(db, "kw_built") is None


@pytest.mark.parametrize("left", [as_sift_1_9_left_it, as_sift_1_8_left_it])
def test_an_older_store_gets_the_postings_tables_unbuilt(corpus, left):
    catalog, _ = corpus
    db = catalog.store("notes")
    if left is as_sift_1_8_left_it:
        as_sift_1_9_left_it(db)
    left(db)
    catalog.release(catalog.get("notes").id)
    db = catalog.store("notes")
    assert db.execute("PRAGMA user_version").fetchone() == (store.SCHEMA_VERSION,)
    assert tables(db) >= POSTINGS_TABLES | {"basenames"}
    assert store.counts(db) == held(db)
    assert store.get_meta(db, "kw_built") is None
    assert not store.basenames_listed(db)


def test_a_store_of_sift_1_10_gets_the_order_of_its_postings(corpus):
    # Layout 9 numbers the postings' chunks in an order of their own (see postings): a store of 1.10 gets its table,
    # empty, which is the order the chunks were indexed in, so its postings stay as they are.
    catalog, _ = corpus
    db = catalog.store("notes")
    db.execute("DROP TABLE kw_order")
    db.execute("PRAGMA user_version = 8")
    db.commit()
    catalog.release(catalog.get("notes").id)
    db = catalog.store("notes")
    assert db.execute("PRAGMA user_version").fetchone() == (store.SCHEMA_VERSION,)
    assert "kw_order" in tables(db)
    assert postings.ready(db)


def test_a_store_upgraded_meanwhile_is_upgraded_once(corpus):
    # Two sifts open a store of 1.9 at once; the second finds the tables there once it has the store.
    db = corpus[0].store("notes")
    before = tables(db)
    store._add_tables(db)  # what the second runs, after the first has finished
    assert tables(db) == before


@pytest.mark.parametrize(
    "left, layout, lacking",
    [(as_sift_1_9_left_it, 7, POSTINGS_TABLES | {"basenames"}), (as_sift_1_12_left_it, 9, {"basenames"})],
)
def test_an_upgrade_stopped_leaves_the_store_as_it_was(corpus, monkeypatch, left, layout, lacking):
    catalog, _ = corpus
    left(catalog.store("notes"))
    catalog.release(catalog.get("notes").id)
    real = store.BASENAMES
    monkeypatch.setattr(store, "BASENAMES", (*real[:-1], "CREATE TRIGGER broken"))  # fails at the last statement
    with pytest.raises(sqlite3.OperationalError):
        catalog.store("notes")
    monkeypatch.setattr(store, "BASENAMES", real)
    raw = sqlite3.connect(catalog.get("notes").store)
    assert raw.execute("PRAGMA user_version").fetchone() == (layout,)
    assert not tables(raw) & lacking
    raw.close()
    assert tables(catalog.store("notes")) >= lacking  # upgraded on the next start


def dirty(db):
    return dict(db.execute("SELECT chunk, change FROM kw_dirty"))


def chunks_of(db, path):
    return {c for (c,) in db.execute("SELECT c.id FROM chunks c JOIN docs d ON d.id = c.doc WHERE d.path = ?", (path,))}


@pytest.mark.usefixtures("unfollowed")
def test_changes_are_marked_once_the_postings_are_built(corpus):
    # The postings follow FTS5, whose entries change only through store.keywords: every chunk whose entries it adds or
    # deletes is marked, in the same transaction, until the postings have it too.
    catalog, root = corpus
    db = catalog.store("notes")
    db.execute("DELETE FROM meta WHERE key = 'kw_built'")
    db.commit()
    change_file(catalog, root)
    assert dirty(db) == {}  # not built: nothing to follow
    store.set_meta(db, "kw_built", 0)
    db.commit()
    old = chunks_of(db, "rl.md")
    change_file_again(catalog, root)
    new = chunks_of(db, "rl.md")
    marked = dirty(db)
    assert set(marked) == old | new
    assert min(marked[c] for c in new) > max(marked[c] for c in old)  # numbered: the later the change, the higher
    assert int(store.get_meta(db, "kw_change")) == max(marked.values())


def change_file_again(catalog, root):
    write(root / "rl.md", "# Reward shaping\n\nWritten a third time.\n")
    indexing.index(catalog)


def test_marks_go_back_with_a_change_undone(corpus):
    db = corpus[0].store("notes")
    store.set_meta(db, "kw_built", 0)
    db.commit()
    store.delete_docs(db, list(doc_ids(db).values()))
    assert dirty(db)
    db.rollback()
    assert dirty(db) == {}


@pytest.mark.usefixtures("unfollowed")
def test_titles_read_again_mark_their_chunks(corpus):
    # retitle changes a document's title in place: its chunks keep their ids, their entries change.
    catalog, _ = corpus
    db = catalog.store("notes")
    store.set_meta(db, "kw_built", 0)
    db.commit()
    before = dict(db.execute("SELECT path, title FROM docs"))
    indexing.retitle(catalog, "notes", "name")
    retitled = [path for path, title in db.execute("SELECT path, title FROM docs") if before[path] != title]
    assert retitled
    assert set(dirty(db)) == set().union(*(chunks_of(db, path) for path in retitled))


# --- the last parts of the documents' paths ---------------------------------------


def listed(db):
    """What `basenames` holds: the last part of a path, and the path, of each row."""
    return sorted(db.execute("SELECT b.name, d.path FROM basenames b LEFT JOIN docs d ON d.id = b.doc"))


def last_parts(db):
    """The last part of every document's path, and the path: what `basenames` is to hold."""
    return sorted((path.rsplit("/", 1)[-1], path) for (path,) in db.execute("SELECT path FROM docs"))


def test_the_last_part_is_what_follows_the_last_slash():
    # SQLite computes it, in the triggers, whatever a path holds.
    rng = random.Random(3)
    letters = ["/", "/", "a", "B", ".", " ", "%", "_", "'", '"', "\\", "\t", "ä", "Ä", "é", "日"]
    paths = ["", "/", "//", "a", "a/", "/a", "a/\U0001f600"]
    paths += ["".join(rng.choices(letters, k=rng.randrange(1, 12))) for _ in range(500)]
    db = sqlite3.connect(":memory:")
    try:
        for path in paths:
            last = db.execute(f"SELECT {store.LAST_PART.format(':path')}", {"path": path}).fetchone()
            assert last == (path.rsplit("/", 1)[-1],), path
    finally:
        db.close()


def test_a_new_store_lists_the_last_part_of_every_path(corpus):
    # A file's name or a record's id: what a document is looked up by when its id is shortened (search.Index._rows).
    catalog, root = corpus
    write(root / "data.jsonl", '{"id": "r/1", "text": "a record"}\n{"id": "end/", "text": "another"}\n')
    indexing.index(catalog)
    db = catalog.store("notes")
    assert listed(db) == last_parts(db)
    assert {("brush.md", "robots/brush.md"), ("1", "data/r/1"), ("", "data/end/")} <= set(listed(db))
    assert store.basenames_listed(db)


def move_document(catalog, root):
    db = catalog.store("notes")
    db.execute("UPDATE docs SET path = 'moved/Elsewhere.md' WHERE path = 'rl.md'")
    db.commit()


@pytest.mark.parametrize("change", [add_file, change_file, remove_file, delete_document, move_document])
def test_a_store_keeps_the_last_parts_through_every_change(corpus, change):
    # By triggers, in the transaction of the change: no document is without its row, and no row without its document.
    catalog, root = corpus
    change(catalog, root)
    db = catalog.store("notes")
    assert listed(db) == last_parts(db)
    assert listed(db)


def test_last_parts_go_back_with_a_change_undone(corpus):
    db = corpus[0].store("notes")
    before = listed(db)
    store.delete_docs(db, list(doc_ids(db).values()))
    assert listed(db) == []
    db.rollback()
    assert listed(db) == before == last_parts(db)


@pytest.fixture
def older(corpus):
    """The corpus in a store as sift 1.12 left it, opened by this sift: (catalog, its folder)."""
    catalog, root = corpus
    as_sift_1_12_left_it(catalog.store("notes"))
    catalog.release(catalog.get("notes").id)
    return catalog, root


def test_a_store_of_sift_1_12_gets_the_table_for_sift_index_to_fill(older, monkeypatch):
    catalog, root = older
    db = catalog.store("notes")
    assert db.execute("PRAGMA user_version").fetchone() == (store.SCHEMA_VERSION,)
    assert listed(db) == []
    assert not store.basenames_listed(db)
    monkeypatch.setattr(store, "list_basenames", lambda *args: None)  # `sift index` has not listed them yet
    add_file(catalog, root)
    assert listed(db) == [("new.md", "new.md")]  # kept from now on, whoever writes


def test_sift_index_lists_the_last_parts_of_an_older_store_once(older, capsys):
    catalog, _ = older
    capsys.readouterr()
    indexing.index(catalog)
    db = catalog.store("notes")
    assert listed(db) == last_parts(db)
    assert store.basenames_listed(db)
    assert store.get_meta(db, "basenames_to") is None
    assert "notes: listing the last part of each document's path, once" in capsys.readouterr().err
    indexing.index(catalog)
    assert "listing" not in capsys.readouterr().err


def test_an_older_store_without_documents_is_listed_without_a_word(tmp_path, capsys):
    db = store.connect(tmp_path / "a.sqlite", 1, create=True)
    as_sift_1_12_left_it(db)
    db.close()
    db = store.connect(tmp_path / "a.sqlite", 1)
    store.list_basenames(db, "a")
    assert store.basenames_listed(db)
    assert capsys.readouterr().err == ""


def counted(monkeypatch, stop_after=None):
    """The batches listed from now on, as each says how far it got; Ctrl-C stops the listing after `stop_after`."""
    ran, batch = [], store._list_batch

    def list_batch(db):
        if len(ran) == stop_after:
            raise KeyboardInterrupt
        ran.append(batch(db))
        return ran[-1]

    monkeypatch.setattr(store, "_list_batch", list_batch)
    return ran


def test_listing_stopped_goes_on_where_it_stopped(older, monkeypatch):
    # Tens of millions of documents take a while: each batch is a transaction of its own that records how far it got,
    # so another sift waits one batch at most, and a listing stopped is not begun again.
    catalog, root = older
    db = catalog.store("notes")
    monkeypatch.setattr(store, "BASENAMES_BATCH", 2)
    with monkeypatch.context() as stopped:
        counted(stopped, stop_after=1)
        with pytest.raises(KeyboardInterrupt):
            store.list_basenames(db, "notes")
    assert listed(db) == [("plain.txt", "plain.txt"), ("rl.md", "rl.md")]  # the first two by path
    assert not store.basenames_listed(db)
    with monkeypatch.context() as meanwhile:  # another sift changes documents, before and after where it stopped
        meanwhile.setattr(store, "list_basenames", lambda *args: None)
        write(root / "a.md", "# Before\n")
        write(root / "z.md", "# After\n")
        (root / "rl.md").unlink()
        indexing.index(catalog)
    ran = counted(monkeypatch)
    indexing.index(catalog)
    assert ran == [4, None]  # robots/brush.md and z.md, then none left: not the first two again
    assert listed(db) == last_parts(db)
    assert store.basenames_listed(db)


def test_a_batch_stopped_leaves_nothing_of_itself(older, monkeypatch):
    # Its rows and how far it got are one transaction: whatever stops it, the next batch begins where the last ended.
    catalog, _ = older
    db = catalog.store("notes")
    monkeypatch.setattr(store, "BASENAMES_BATCH", 2)
    assert store._list_batch(db) == 2

    def stopped(db, key, value):  # Ctrl-C between a batch's rows and the record of them
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(store, "set_meta", stopped)
        with pytest.raises(KeyboardInterrupt):
            store._list_batch(db)
    assert not db.in_transaction
    assert listed(db) == [("plain.txt", "plain.txt"), ("rl.md", "rl.md")]
    assert store.get_meta(db, "basenames_to") == '[2, "rl.md"]'
    assert [store._list_batch(db), store._list_batch(db)] == [3, None]
    assert listed(db) == last_parts(db)


def test_two_sifts_listing_at_once_take_turns(older, monkeypatch):
    catalog, _ = older
    monkeypatch.setattr(store, "BASENAMES_BATCH", 1)
    one, other = catalog.store("notes"), store.database(catalog.get("notes").store)
    assert [store._list_batch(db) for db in (one, other, one, other, one)] == [1, 2, 3, None, None]  # each batch once
    assert listed(one) == last_parts(one)
    assert store.basenames_listed(one)
    assert store.basenames_listed(other)


def test_a_batch_another_sift_listed_meanwhile_is_left_to_it(older, monkeypatch):
    # A batch is found before its transaction, so that the store is held no longer than writing it takes: the other
    # sift may have listed it by then, and further.
    catalog, _ = older
    monkeypatch.setattr(store, "BASENAMES_BATCH", 1)
    one, other = catalog.store("notes"), store.database(catalog.get("notes").store)

    def meanwhile(sql):  # one has found its batch, and is about to begin its transaction
        if sql.startswith("BEGIN"):
            one.set_trace_callback(None)
            assert [store._list_batch(other), store._list_batch(other)] == [1, 2]

    one.set_trace_callback(meanwhile)
    assert store._list_batch(one) == 2  # as far as the other got: on after it
    assert listed(one) == [("plain.txt", "plain.txt"), ("rl.md", "rl.md")]
    assert [store._list_batch(one), store._list_batch(one)] == [3, None]
    assert listed(one) == last_parts(one)


def test_documents_come_and_go_while_two_sifts_list(corpus, monkeypatch):
    # However their transactions fall: in the end every document has its row, and no row is without its document.
    catalog, _ = corpus
    notes, db = catalog.get("notes"), catalog.store("notes")
    add = "INSERT INTO docs(source, path, title, meta, text) SELECT min(id), ?, 'x', '{}', 'x' FROM sources"
    db.executemany(add, [(f"bulk/{i:03}.md",) for i in range(200)])
    db.commit()
    as_sift_1_12_left_it(db)
    catalog.release(notes.id)
    monkeypatch.setattr(store, "BASENAMES_BATCH", 7)
    listing, changing = [store.connect(notes.store, notes.id) for _ in range(2)], store.connect(notes.store, notes.id)

    def lists(db):
        while store._list_batch(db) is not None:
            pass

    def changes():
        for i in range(0, 200, 4):
            changing.execute(add, (f"{i % 7}/new/{i}.md",))
            changing.execute("DELETE FROM docs WHERE path = ?", (f"bulk/{i:03}.md",))
            changing.commit()

    with ThreadPoolExecutor(3) as threads:
        for work in [threads.submit(changes), *(threads.submit(lists, db) for db in listing)]:
            work.result()
    db = catalog.store("notes")
    assert listed(db) == last_parts(db)
    assert len(listed(db)) == 203
    assert store.basenames_listed(db)


# --- version -------------------------------------------------------------------


def test_delete_marks_vectors_changed(corpus):
    db = corpus[0].store("notes")
    before = store.get_meta(db, "version")
    store.delete_docs(db, [])
    assert store.get_meta(db, "version") == before
    store.delete_docs(db, list(doc_ids(db).values())[:1])
    assert store.get_meta(db, "version") != before


def epoch(db):
    return store.get_meta(db, "codes_epoch")


def test_codes_appended_past_the_last_keep_their_epoch(embedded):
    # A running server reads only the codes past the last it holds, as long as this stays: `sift embed` writes its
    # slabs in the order of the chunks, each past the codes before.
    db = embedded[0].store("notes")
    last, before = db.execute("SELECT max(chunk) FROM codes").fetchone()[0], epoch(db)
    vector = np.ones((1, 64), np.float16)
    store.put_vectors(db, [last + 1], vector)
    assert epoch(db) == before
    store.put_vectors(db, [last + 1], -vector)  # embedded again: a code it holds changes
    assert epoch(db) != before
    again = epoch(db)
    store.put_vectors(db, [last - 1], vector)  # one before the last
    assert epoch(db) != again


@pytest.mark.parametrize("dropped", [True, False])
def test_vectors_dropped_change_the_epoch(embedded, dropped):
    catalog, root = embedded
    db = catalog.store("notes")
    before = epoch(db)
    if dropped:
        store.delete_docs(db, [doc_ids(db)["rl.md"]])
    else:
        write(root / "new.md", "# New\n\nNot embedded yet.\n")
        indexing.index(catalog)
        store.delete_docs(db, [doc_ids(db)["new.md"]])  # it had no vector: no code changes
    assert (epoch(db) != before) == dropped


def test_every_deletion_of_vectors_goes_through_the_store():
    # One place keeps the epoch: a deletion elsewhere would leave a running server holding codes of vectors gone.
    package = Path(store.__file__).parent
    found = [
        f"{path.relative_to(package)}:{n}"
        for path in sorted(package.rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"(delete\s+from|drop\s+table|truncate)\s+(main\.)?vectors\b", line, re.IGNORECASE)
    ]
    assert len(found) == 1
    assert found[0].startswith("store.py:")


def test_codes_are_read_past_a_chunk(embedded):
    db = embedded[0].store("notes")
    ids, bits = store.read_codes(db)
    tail_ids, tail_bits = store.read_codes(db, after=int(ids[0]))
    assert (tail_ids.tolist(), tail_bits.tolist()) == (ids[1:].tolist(), bits[1:].tolist())
    assert store.read_codes(db, after=int(ids[-1])) is None
    assert store.read_codes(db, after=int(ids[0]), most=1)[0].tolist() == ids[1:2].tolist()


def test_codes_read_past_a_chunk_take_the_room_of_those_found(embedded):
    # A server reads the codes past the last it holds on every search while it reads all anew, at most as many as
    # the store counts: room for 29 million was taken each time, for a few.
    db = embedded[0].store("notes")
    ids, _ = store.read_codes(db)
    tracemalloc.start()
    try:
        store.read_codes(db, after=int(ids[0]), most=10_000_000)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000


def test_touch_changes_version_within_one_clock_tick(corpus, monkeypatch):
    db = corpus[0].store("notes")
    monkeypatch.setattr(time, "time_ns", lambda: 1)  # before Python 3.13 the Windows clock ticks every 15.6 ms
    store.touch(db)
    before = store.get_meta(db, "version")
    store.touch(db)
    assert store.get_meta(db, "version") != before


# --- the write-ahead log ---------------------------------------------------------


def wal_size(path):
    log = Path(f"{path}-wal")
    return log.stat().st_size if log.exists() else 0


def test_a_write_ahead_log_grown_while_read_shrinks_back(tmp_path, monkeypatch):
    # A reader's snapshot keeps every write after it in the log (a build's workers read for minutes): once it ends,
    # the log goes back to its limit, though other connections keep the store open.
    monkeypatch.setattr(store, "WAL_LIMIT", 64 * 1024)
    path = tmp_path / "a.sqlite"
    db = store.connect(path, 1, create=True)
    reader = store.connect(path, 1)
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM meta").fetchone()
    for i in range(30):
        store.set_meta(db, f"k{i}", "x" * 200_000)
        db.commit()
    assert wal_size(path) > 5_000_000  # past SQLite's checkpoint of 1000 pages, which can't empty it while read
    reader.rollback()
    for i in range(3):
        store.set_meta(db, "after", i)
        db.commit()
    assert wal_size(path) <= store.WAL_LIMIT
    reader.close()


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


def test_only_damage_counts_as_a_damaged_store(tmp_path):
    # A store locked by another program, or a disk full, is no reason to tell the user to forget the collection.
    damaged = tmp_path / "damaged.sqlite"
    damaged.write_bytes(b"not a database, but sixteen bytes and more")
    db = sqlite3.connect(damaged)
    with pytest.raises(sqlite3.DatabaseError) as found:
        db.execute("SELECT * FROM sqlite_master").fetchall()
    db.close()
    assert store.damage("notes", found.value).startswith("notes: its index is damaged (file is not a database)")
    locked = sqlite3.OperationalError("database is locked")
    locked.sqlite_errorcode = sqlite3.SQLITE_BUSY
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        store.damage("notes", locked)


def test_what_a_file_waits_for_is_found_by_its_file_without_a_scan(tmp_path):
    # Every file stored asks: a scan of a million rows took 16 ms each.
    db = store.connect(tmp_path / "s.sqlite", 1, create=True)
    store.wait(db, "data/a.md", 7)
    plan = " ".join(row[-1] for row in db.execute("EXPLAIN QUERY PLAN DELETE FROM waiting WHERE source = 7"))
    db.close()
    assert "USING INDEX waiting_source" in plan


def test_a_marking_once_that_fails_otherwise_than_on_a_store_sift_may_only_read_says_so(tmp_path):
    db = store.connect(tmp_path / "s.sqlite", 1, create=True)

    def failing(db):
        raise sqlite3.OperationalError("disk I/O error")

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        store._once(db, "some_marking", failing)
    assert store.get_meta(db, "some_marking") is None  # tried again next time
    db.close()


# --- the chunks' texts -----------------------------------------------------------


def _made_up(rng, n):
    """Text of what stores hold: accents composed and not, a character beyond the BMP, line ends, U+2028."""
    letters = ["a", "b", "c", " ", " ", "\n", "\r\n", "\u00e9", "e\u0301", "\u65e5", "\U0001f600", "\u2028", "'"]
    return "".join(rng.choice(letters) for _ in range(n))


def test_chunks_are_cut_in_python_as_the_view_cuts_them(tmp_path, monkeypatch):
    # The keyword index's entries are deleted by the values they were added with, and older sifts added them from the
    # chunk_text view: rows cut from a text read once must be the view's, to the character. A NUL, at which SQLite's
    # substr stops, is in the texts of stores written before sift 1.12.
    monkeypatch.setattr(store, "CUT", 7)  # a document's chunks lie across the batches they are cut in
    rng, db = random.Random(5), store.connect(tmp_path / "a.sqlite", 1, create=True)
    for n in range(40):
        text = _made_up(rng, rng.randint(0, 300))
        if n % 4 == 0:
            at = rng.randint(0, len(text))
            text = f"{text[:at]}\x00{text[at:]}"
        meta = {
            "title": "left out",
            "authors": ["Ann Lee", "Bo"],
            "year": 2000 + n,
            "none": None,
            "in": {"id": f"x{n}"},
        }
        row = (f"d{n}", _made_up(rng, 5), json.dumps(meta if n % 3 else {}), text)
        doc = db.execute(
            "INSERT INTO docs(source, path, hash, title, meta, text) VALUES(1, ?, '', ?, ?, ?)", row
        ).lastrowid
        spans = [(doc, seq, rng.randint(0, len(text) + 2), rng.randint(0, 80)) for seq in range(rng.randint(0, 5))]
        db.executemany("INSERT INTO chunks(doc, seq, line_start, line_end, pos, len) VALUES(?, ?, 1, 1, ?, ?)", spans)
    db.execute("INSERT INTO chunks(doc, seq, line_start, line_end, pos, len) VALUES(999, 0, 1, 1, 0, 5)")  # no document
    view = db.execute("SELECT id, text, title, meta FROM chunk_text ORDER BY id").fetchall()
    assert len(view) > 60
    assert list(store.chunk_rows(db)) == view
    assert list(store.chunk_rows(db, chunks=[row[0] for row in view[::3]] * 2)) == view[::3]  # each named twice
    odd = [doc for (doc,) in db.execute("SELECT id FROM docs WHERE id % 2")]
    theirs = {chunk for (chunk,) in db.execute("SELECT id FROM chunks WHERE doc % 2")}
    assert list(store.chunk_rows(db, docs=odd)) == [row for row in view if row[0] in theirs]
    spans = db.execute("SELECT id, doc, pos, len FROM chunks").fetchall()
    assert store.texts(db, spans) == {row[0]: row[1] for row in view}
    monkeypatch.setattr(store, "LONG", 2)  # some documents' entries through the view, the others' cut in Python
    store.fill(db, "fts")
    assert keywords_agree(db)
    store.fill(db, "fts", delete=True)
    db.execute("CREATE VIRTUAL TABLE temp.terms USING fts5vocab(main, fts, row)")
    assert db.execute("SELECT * FROM temp.terms").fetchall() == []
    db.close()


def test_a_document_with_a_nul_from_an_older_sift_goes_with_all_its_keyword_entries(tmp_path):
    # Its entries were added as the view read them, each chunk up to the NUL: deleted by the whole chunk, the words
    # after the NUL would be taken out of an index that never held them.
    db = store.connect(tmp_path / "a.sqlite", 1, create=True)
    text = "alpha beta\x00gamma delta"
    doc = db.execute(
        "INSERT INTO docs(source, path, hash, title, meta, text) VALUES(1, 'a.md', '', 'A', '{}', ?)", (text,)
    ).lastrowid
    spans = [(doc, 0, 0, len(text)), (doc, 1, 6, 9), (doc, 2, 11, 5)]
    db.executemany("INSERT INTO chunks(doc, seq, line_start, line_end, pos, len) VALUES(?, ?, 1, 1, ?, ?)", spans)
    db.execute("INSERT INTO fts(rowid, title, text, meta) SELECT id, title, text, meta FROM chunk_text")  # as sift 1.11
    assert db.execute("SELECT count(*) FROM fts WHERE fts MATCH 'beta'").fetchone() == (2,)
    assert db.execute("SELECT count(*) FROM fts WHERE fts MATCH 'gamma'").fetchone() == (0,)
    store.delete_docs(db, [doc])
    db.execute("CREATE VIRTUAL TABLE temp.terms USING fts5vocab(main, fts, row)")
    assert db.execute("SELECT * FROM temp.terms").fetchall() == []
    db.execute("INSERT INTO fts(fts) VALUES('integrity-check')")
    db.close()


SHAPES = [  # front matter as files and records hold it: SQLite writes its values, in its order, not Python
    {"title": "T", "ratio": 0.1, "big": 2**63 + 5, "neg": -(2**64), "exp": 1e300, "yes": True, "no": False},
    {"authors": [{"name": "Ann", "ids": [1, 2.5, None]}, "Bo"], "deep": {"a": {"b": ["x", {"c": "y"}]}}},
    {"nan": "NaN", "inf": "Infinity", "tiny": 5e-324, "zero": -0.0, "unicode": "Zürich 日本 \U0001f600 é"},
    {"nul": "a\ufffdb", "empty": "", "list": [], "obj": {}, "num": "0042", "title": "kept out"},
]


def test_keyword_rows_of_long_documents_are_the_views_for_front_matter_of_every_shape(tmp_path, monkeypatch):
    # An entry deleted with other values than it was added with damages FTS5's index without an error, and stores in
    # use hold entries the view wrote: rows cut in Python must carry its title and meta to the character.
    monkeypatch.setattr(store, "LONG", 0)  # every document long: its rows cut in Python
    db = store.connect(tmp_path / "a.sqlite", 1, create=True)
    for n, meta in enumerate(SHAPES):
        text = f"alpha {n}\x00beta" if n == 3 else f"alpha {n} beta gamma"
        row = (f"d{n}", meta.get("title", f"D{n}"), json.dumps(meta, ensure_ascii=n % 2 == 0), text)
        doc = db.execute("INSERT INTO docs(source, path, hash, title, meta, text) VALUES(1, ?, '', ?, ?, ?)", row)
        spans = [(doc.lastrowid, s, s * 3, 9) for s in range(3)]
        db.executemany("INSERT INTO chunks(doc, seq, line_start, line_end, pos, len) VALUES(?, ?, 1, 1, ?, ?)", spans)
    view = db.execute("SELECT id, text, title, meta FROM chunk_text ORDER BY id").fetchall()
    assert list(store.chunk_rows(db)) == view
    db.execute("INSERT INTO fts(rowid, title, text, meta) SELECT id, title, text, meta FROM chunk_text")  # as 1.12
    docs = [doc for (doc,) in db.execute("SELECT id FROM docs")]
    store.keywords(db, docs, add=False)  # deleted by rows cut in Python
    db.execute("CREATE VIRTUAL TABLE temp.terms USING fts5vocab(main, fts, row)")
    assert db.execute("SELECT * FROM temp.terms").fetchall() == []
    store.keywords(db, docs)
    assert keywords_agree(db)
    store.delete_docs(db, docs)
    assert db.execute("SELECT * FROM temp.terms").fetchall() == []
    db.execute("INSERT INTO fts(fts) VALUES('integrity-check')")
    db.close()


@pytest.fixture
def statements(monkeypatch):
    """Every statement sift runs on any store it opens, FTS5's own on its tables and its content among them."""
    ran = []

    def traced(opening):
        def opened(*args, **kwargs):
            db = opening(*args, **kwargs)
            db.set_trace_callback(ran.append)
            return db

        return opened

    for opener in ("database", "read_only"):
        monkeypatch.setattr(store, opener, traced(getattr(store, opener)))
    return ran


def _lived(tmp_path, sections):
    """A note of so many sections, a chunk each, indexed, embedded, searched, changed, retitled and removed."""
    root, catalog = tmp_path / f"notes{sections}", Catalog(tmp_path / f"home{sections}")
    note = "".join(f"## Section {s}\n\n{' '.join(f'w{s}x{i}' for i in range(250))}\n\n" for s in range(sections))
    write(root / "long.md", note)
    catalog.add(root, "long")
    indexing.index(catalog)
    assert store.counts(catalog.store("long"))["chunks"] == sections
    embed.embed_pending(catalog)
    index = Index(catalog)
    assert len(index.search('"w1x1 w1x2"', all_passages=True)) == 1
    assert len(index.search("w0x1 OR w1x1 OR w2x1", all_passages=True)) == 3
    write(root / "long.md", note + "## One more\n\nw1x1 w1x2 again.\n")
    assert index.reindex(["long/long.md"])[0]["embedded"] == sections + 1
    assert len(index.search('"w1x1 w1x2"', all_passages=True)) == 2
    assert indexing.retitle(catalog, "long", "name") == (1, sections)
    (root / "long.md").unlink()
    indexing.index(catalog)
    assert store.counts(catalog.store("long"))["chunks"] == 0
    catalog.close()


def test_a_long_documents_text_is_read_once_however_many_chunks_it_has(tmp_path, fake_models, statements):
    # SQLite's substr loads a document's whole text and walks to the position for every chunk it cuts. An 8 MB file's
    # 3,310 chunks took 4 s to store with their keyword entries and 20 s to read back for embedding, 13 and 17 times
    # what a 2 MB file took; an 80 MB file took 11 minutes to store. The one statement left that cuts in SQL, FTS5's
    # own reads of its content counted, leaves long documents out, and a note of forty chunks has its text read as
    # often as one of twelve.
    read = []
    for sections in (12, 40):
        del statements[:]
        _lived(tmp_path, sections)
        cutting = [s for s in statements if ("chunk_text" in s or "c.pos + 1" in s) and "CREATE V" not in s]
        assert cutting  # the view's definition and its table's aside: the store's statement for short documents
        assert all("WHERE NOT EXISTS (SELECT 1 FROM chunks k WHERE k.doc = c.doc" in s for s in cutting)
        read.append(sum("d.text" in s for s in statements))
    assert read[0] == read[1] > 0


def test_only_the_chunks_of_long_documents_are_cut_in_python(tmp_path, statements):
    # Cut in Python, a row per chunk, the keyword entries of the 300,000 records of a JSON Lines file took a tenth
    # longer to store than through the view's one statement: that is for documents of more than LONG chunks, whose
    # text the view would load once per chunk.
    root, catalog = tmp_path / "notes", Catalog(tmp_path / "home")
    section = "## Section {}\n\n" + " ".join(f"w{i}" for i in range(400)) + "\n\n"
    write(root / "long.md", "".join(section.format(n) for n in range(store.LONG + 1)))
    for name in "abc":
        write(root / f"{name}.md", "".join(section.format(n) for n in range(2)))
    catalog.add(root, "notes")
    indexing.index(catalog)
    db = catalog.store("notes")
    assert store.counts(db)["chunks"] == store.LONG + 1 + 3 * 2
    assert sum(s.startswith("INSERT INTO fts(rowid, text, title, meta) VALUES") for s in statements) == store.LONG + 1
    assert keywords_agree(db)
    for path in root.iterdir():
        path.unlink()
    indexing.index(catalog)
    assert sum("VALUES('delete'" in s for s in statements) == store.LONG + 1
    db.execute("CREATE VIRTUAL TABLE temp.terms USING fts5vocab(main, fts, row)")
    assert db.execute("SELECT * FROM temp.terms").fetchall() == []
    catalog.close()
