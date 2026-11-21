"""`sift index`: scanning folders, reading new and changed files in worker processes, and keeping the index in
step with the files, incrementally."""

import multiprocessing
import os
import shutil
import sqlite3
import stat
import time

import pytest

from sift import embed, indexing, readers, store
from sift.catalog import Catalog
from sift.ignore import Ignore
from sift.search import Index

from . import workers
from .conftest import index_contents, keywords_agree, only, write
from .test_documents import email_file


def doc_ids(catalog):
    """The documents of every collection, by path."""
    return {path: i for c in catalog.collections() for path, i in catalog.store(c).execute("SELECT path, id FROM docs")}


def test_walk_skips_hidden_and_vendor_dirs(notes):
    found = sorted(os.path.relpath(p, notes).replace(os.sep, "/") for p, _, _ in indexing.walk(str(notes)))
    assert found == ["plain.txt", "rl.md", "robots/brush.md"]


def test_walk_survives_missing_root_and_unstatable_files(tmp_path, monkeypatch):
    assert list(indexing.walk(str(tmp_path / "missing"))) == []

    class Entry:
        name, path = "broken.md", "broken.md"

        def is_dir(self, follow_symlinks):
            return False

        def stat(self):
            raise FileNotFoundError

    class Entries(list):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(indexing.os, "scandir", lambda path: Entries([Entry()]))
    assert list(indexing.walk("anywhere")) == []


def test_a_file_named_like_an_extension_has_none(corpus, capsys):
    # As for `reindex`: `.md` is a name starting with a dot, not a Markdown file without one. `sift index` listed it,
    # found no reader for it, and stopped with "error: ''"; an app that names a note by its empty title writes one.
    catalog, root = corpus
    write(root / ".md", "# Nameless\n")
    write(root / ".notes.md", "# A note whose name starts with a dot\n")
    found = {os.path.relpath(p, root) for p, _, _ in indexing.walk(str(root))}
    assert ".notes.md" in found
    assert ".md" not in found
    indexing.index(catalog)
    assert ".notes.md" in doc_ids(catalog)


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows'")
def test_walk_follows_no_junction(notes, tmp_path):
    # A junction to the collection's own folder made every file come again, 64 times; one to elsewhere took in files
    # of no collection.
    import _winapi

    write(tmp_path / "elsewhere" / "outside.md", "# Outside\n")
    _winapi.CreateJunction(str(notes), str(notes / "loop"))
    _winapi.CreateJunction(str(tmp_path / "elsewhere"), str(notes / "away"))
    found = sorted(os.path.relpath(p, notes).replace(os.sep, "/") for p, _, _ in indexing.walk(str(notes)))
    assert found == ["plain.txt", "rl.md", "robots/brush.md"]


def orphans(db):
    return db.execute("SELECT count(*) FROM docs WHERE source NOT IN (SELECT id FROM sources)").fetchone()[0]


def reindexed_meanwhile(monkeypatch, catalog, paths, then=lambda: None):
    """Have another sift, a server, run then() and reindex `paths` while `sift index` reads its first file."""
    real, done = indexing._read_scans, []

    def meanwhile(*args, **kwargs):
        if not done:
            done.append(1)
            then()
            other = Catalog(catalog.home)
            indexing.reindex(other, paths)
            other.close()
        return real(*args, **kwargs)

    monkeypatch.setattr(indexing, "_read_scans", meanwhile)


def test_a_note_a_server_reindexed_meanwhile_is_indexed_once(corpus, monkeypatch):
    # An agent wrote a note and reindexed it while `sift index` read the folder: its source, new when scanned, was
    # there by the time `sift index` stored it, which then failed for this collection and every one after it.
    catalog, root = corpus
    write(root / "new.md", "# New\n\nWritten by an agent.\n")
    reindexed_meanwhile(monkeypatch, catalog, ["notes/new.md"])
    indexing.index(catalog)
    db = catalog.store("notes")
    assert db.execute("SELECT count(*) FROM docs WHERE path = 'new.md'").fetchone() == (1,)
    assert orphans(db) == 0


def test_a_note_a_server_removed_meanwhile_leaves_no_documents_behind(corpus, monkeypatch):
    # Deleted and reindexed while `sift index` read it, a note was stored under its source gone: no later `sift index`
    # removed its documents, and searches found them for good.
    catalog, root = corpus
    write(root / "rl.md", "# Reward shaping\n\nWritten again.\n")
    real = indexing._read_scans
    reindexed_meanwhile(monkeypatch, catalog, ["notes/rl.md"], then=(root / "rl.md").unlink)
    indexing.index(catalog)
    db = catalog.store("notes")
    assert orphans(db) == 0
    monkeypatch.setattr(indexing, "_read_scans", real)  # not undo(): that would undo the fixtures' too
    indexing.index(catalog)
    assert "rl.md" not in doc_ids(catalog)


def lock_free(path):
    """Whether another sift could take the store's write lock now, without waiting."""
    other = sqlite3.connect(path, timeout=0, isolation_level=None)
    try:
        other.execute("BEGIN IMMEDIATE")
        other.execute("ROLLBACK")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        other.close()


def test_sift_index_holds_no_write_lock_while_it_reads_files(corpus, monkeypatch):
    # The first write of a batch, a removal too, took the store's lock until the batch's end: `sift embed` or a
    # server's reindex beside it waited while every file of the batch was read, and gave up after 30 seconds.
    catalog, root = corpus
    for i in range(3):
        write(root / f"new{i}.md", f"# New {i}\n")
    (root / "plain.txt").unlink()  # removed first, before any file is read
    path, real, free = catalog.get("notes").store, readers.load, []
    monkeypatch.setattr(readers, "load", lambda *args: free.append(lock_free(path)) or real(*args))
    indexing.index(catalog)
    assert free == [True] * 3


def test_sift_index_holds_no_write_lock_while_it_reads_a_file_freed_to_be_read(corpus, monkeypatch):
    # Asking which files a freed path let through wrote, and held the lock while they, and the scan's, were read.
    catalog, root = corpus
    write(root / "data.jsonl", '{"id": "a.md", "text": "zebra in the record"}\n')
    write(root / "data" / "a.md", "# A\n\ngiraffe in the real file\n")
    indexing.index(catalog)  # the record keeps the path, the file waits
    write(root / "data.jsonl", '{"id": "b.md", "text": "zebra in another record"}\n')
    write(root / "other.md", "# Other\n")
    path, real, free = catalog.get("notes").store, readers.load, []
    monkeypatch.setattr(readers, "load", lambda *args: free.append(lock_free(path)) or real(*args))
    indexing.index(catalog)
    assert free == [True] * 3  # data.jsonl and other.md, then data/a.md, freed by the first


def test_retitling_holds_the_write_lock_a_batch_at_a_time(corpus, monkeypatch):
    catalog, _ = corpus
    monkeypatch.setattr(indexing, "REFRESH_BATCH", 1)
    path, real, free = catalog.get("notes").store, indexing._reread, []
    monkeypatch.setattr(indexing, "_reread", lambda db, *args: free.append(lock_free(path)) or real(db, *args))
    indexing.retitle(catalog, "notes", "name")
    assert len(free) == 3
    assert all(free)


def test_junctions_are_told_apart_where_python_knows_no_reparse_tags(notes, monkeypatch):
    # stat.IO_REPARSE_TAG_MOUNT_POINT is Windows' alone: elsewhere, asking for it failed every walk of every folder.
    import stat

    if hasattr(stat, "IO_REPARSE_TAG_MOUNT_POINT"):
        assert indexing.MOUNT_POINT == stat.IO_REPARSE_TAG_MOUNT_POINT
    monkeypatch.delattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", raising=False)
    found = sorted(os.path.relpath(p, notes).replace(os.sep, "/") for p, _, _ in indexing.walk(str(notes)))
    assert found == ["plain.txt", "rl.md", "robots/brush.md"]
    assert not indexing._linked(str(notes / "robots"))


def test_a_folder_gone_meanwhile_is_no_junction(tmp_path):
    # Listed, then deleted before it is looked at: nothing to follow, and nothing to fail on.
    assert indexing._junction(str(tmp_path / "gone")) is False
    assert not indexing._linked(str(tmp_path / "gone"))


def test_walk_leaves_out_what_siftignore_names(notes):
    write(notes / ".siftignore", "plain.txt\nrobots/\n")
    write(notes / "robots" / "deep" / "x.md", "# Inside a folder left out\n")
    found = [os.path.relpath(p, notes) for p, _, _ in indexing.walk(str(notes), Ignore.of(notes))]
    assert found == ["rl.md"]


def test_index_follows_siftignore_as_it_changes(corpus, capsys):
    # An app keeps its own files out, a user drafts or exports; and what is left out after all leaves the index.
    catalog, root = corpus
    write(root / ".siftignore", "# drafts and exports\nrobots/\n*.txt\n")
    indexing.index(catalog)
    assert sorted(doc_ids(catalog)) == ["rl.md"]
    assert "2 removed" in capsys.readouterr().err
    write(root / ".siftignore", "robots/\n")
    indexing.index(catalog)
    assert sorted(doc_ids(catalog)) == ["plain.txt", "rl.md"]


def test_a_collection_whose_siftignore_cannot_be_read_is_skipped(corpus, tmp_path, capsys):
    # Indexing all of it would take in the very files the .siftignore keeps out, as an offline OneDrive file can't
    # be read; its documents stay, and the collections after it are indexed still.
    catalog, root = corpus
    write(tmp_path / "other" / "x.md", "# X\n")
    catalog.add(tmp_path / "other", "other")
    (root / ".siftignore").mkdir()
    before = doc_ids(catalog)
    indexing.index(catalog)
    assert "notes: skipped, as its .siftignore can't be read: " in capsys.readouterr().err
    assert doc_ids(catalog).keys() == before.keys() | {"x.md"}


# --- indexing ----------------------------------------------------------------


def test_index_picks_text_and_markdown_only(corpus):
    assert sorted(doc_ids(corpus[0])) == ["plain.txt", "rl.md", "robots/brush.md"]


def test_files_an_older_sift_read_with_replacement_characters_are_read_again_once(tmp_path, home, monkeypatch, capsys):
    # sift 1.14 made U+FFFD of every byte that wasn't UTF-8, as in a note saved as ANSI: unchanged, such a file was
    # never read again, and "Müller" stayed unfindable. The next index reads again the text files, JSON Lines and
    # mails whose text holds one, a batch of documents at a time, and only once.
    root = tmp_path / "c"
    root.mkdir()
    (root / "ansi.txt").write_bytes("Grüezi Müller\n".encode("cp1252"))
    (root / "fine.md").write_bytes(b"# Fine\n\nnothing to read again\n")
    (root / "data.jsonl").write_bytes('{"id": "1", "text": "Zürich"}\n'.encode("cp1252"))
    catalog = Catalog(home)
    catalog.add(root, "c")
    indexing.index(catalog)
    db = only(catalog)
    db.execute("UPDATE docs SET text = replace(text, 'ü', char(65533))")  # as sift 1.14 stored them
    db.execute("DELETE FROM meta WHERE key = 'ansi_read'")
    db.commit()
    monkeypatch.setattr(store, "ANSI_BATCH", 2)  # three documents: two batches
    capsys.readouterr()
    indexing.index(catalog)
    assert dict(db.execute("SELECT path, text FROM docs")) == {
        "ansi.txt": "Grüezi Müller\n",
        "fine.md": "# Fine\n\nnothing to read again\n",
        "data/1": "Zürich",
    }
    said = capsys.readouterr().err
    assert "replacement characters" in said
    assert "3 files, 0 added, 2 changed, 0 removed" in said
    indexing.index(catalog)  # once
    assert "3 files, 0 added, 0 changed, 0 removed" in capsys.readouterr().err
    assert store.get_meta(db, "ansi_read") == "1"


def test_a_look_for_replacement_characters_stopped_goes_on_where_it_stopped(tmp_path, home, monkeypatch, capsys):
    root = tmp_path / "c"
    root.mkdir()
    for name in "abc":
        (root / f"{name}.txt").write_bytes(f"{name} Müller\n".encode("cp1252"))
    catalog = Catalog(home)
    catalog.add(root, "c")
    indexing.index(catalog)
    db = only(catalog)
    db.execute("UPDATE docs SET text = replace(text, 'ü', char(65533))")
    db.execute("DELETE FROM meta WHERE key = 'ansi_read'")
    db.commit()
    monkeypatch.setattr(store, "ANSI_BATCH", 1)
    set_meta, calls = store.set_meta, []

    def stopped_in_the_second_batch(db, key, value):
        calls.append(key)
        if calls.count("ansi_read_to") == 2:
            raise KeyboardInterrupt
        set_meta(db, key, value)

    monkeypatch.setattr(store, "set_meta", stopped_in_the_second_batch)
    with pytest.raises(KeyboardInterrupt):
        indexing.index(catalog)
    monkeypatch.setattr(store, "set_meta", set_meta)
    assert db.execute("SELECT path FROM sources WHERE mtime IS NULL").fetchall() == [("a.txt",)]  # the first batch's
    indexing.index(catalog)
    assert sorted(db.execute("SELECT path, text FROM docs")) == [(f"{n}.txt", f"{n} Müller\n") for n in "abc"]


def test_an_older_store_without_documents_is_looked_through_without_a_word(tmp_path, home, capsys):
    (tmp_path / "c").mkdir()
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    db = only(catalog)
    db.execute("DELETE FROM meta WHERE key = 'ansi_read'")
    db.commit()
    indexing.index(catalog)
    assert "replacement characters" not in capsys.readouterr().err
    assert store.get_meta(db, "ansi_read") == "1"


def test_a_nul_in_a_file_hides_nothing_after_it(corpus):
    # SQLite's substr, which cuts the passages out of a document's text, stops at a NUL: the words after one, in a log
    # padded with them or a note with a pasted control character, went into the keyword index and the embeddings empty.
    catalog, root = corpus
    (root / "padded.txt").write_bytes(b"before the padding\x00\x00\x00 wombat after it\n")
    write(root / "pasted.md", "# Num\x00bers\n\nnumbat in the text\n")
    indexing.index(catalog)
    index = Index(catalog)
    for word, path in (("wombat", "notes/padded.txt"), ("numbat", "notes/pasted.md")):
        hits = index.search(word)
        assert [hit["id"] for hit in hits] == [path]
        assert word in hits[0]["text"]
    assert index.document("notes/pasted.md", fields=["title"])["title"] == "Num\ufffdbers"


@pytest.mark.parametrize("record_first", [True, False])
def test_a_document_left_out_for_its_path_gets_it_once_free(corpus, capsys, record_first):
    # A JSON Lines record (or a mail's attachment) is a document below its file's name, a path a real file may have:
    # the one left out was never read again, even once the path was free; then read again by every run, a PDF
    # converted and a scan read anew each time; then a record left out for a file never came back.
    catalog, root = corpus
    record, real = root / "data.jsonl", root / "data" / "a.md"
    write(real, "# A\n\ngiraffe in the real file\n")
    if not record_first:  # the file indexed first, the record added later
        indexing.index(catalog)
    write(record, '{"id": "a.md", "text": "zebra in the record"}\n')  # else read first, in one run
    indexing.index(catalog)
    assert capsys.readouterr().err.count("left out until it is free") == 1
    holder, waiting = (record, "giraffe") if record_first else (real, "zebra")
    assert Index(catalog).search(waiting) == []
    indexing.index(catalog)
    assert "0 added, 0 changed, 0 removed" in capsys.readouterr().err  # nothing read again meanwhile
    holder.unlink()
    indexing.index(catalog)
    assert [hit["id"] for hit in Index(catalog).search(waiting)] == ["notes/data/a.md"]


def test_a_holder_that_changes_frees_its_path_in_the_same_run(corpus):
    catalog, root = corpus
    write(root / "data.jsonl", '{"id": "a.md", "text": "zebra in the record"}\n')
    write(root / "data" / "a.md", "# A\n\ngiraffe in the real file\n")
    indexing.index(catalog)  # the record first
    write(root / "data.jsonl", '{"id": "b.md", "text": "zebra in another record"}\n')
    indexing.index(catalog)
    assert [hit["id"] for hit in Index(catalog).search("giraffe")] == ["notes/data/a.md"]


def test_only_files_that_wait_for_a_path_are_read_again(corpus, monkeypatch):
    # Every document deleted read again the files named like a folder above it: Project.md, beside Project/ as Obsidian
    # keeps them, at every change in that folder; data.2025.jsonl, whole, at every change of data.jsonl.
    catalog, root = corpus
    write(root / "Project.md", "# Project\n")
    write(root / "Project" / "sub" / "x.md", "# X\n")
    write(root / "Project" / "y.md", "# Y\n")
    write(root / "data.jsonl", '{"id": "1", "text": "one"}\n')
    write(root / "data.2025.jsonl", '{"id": "1", "text": "one of 2025"}\n')
    indexing.index(catalog)
    read, real = [], indexing._loaded
    monkeypatch.setattr(indexing, "_loaded", lambda path, fields: read.append(path) or real(path, fields))
    write(root / "Project" / "sub" / "x.md", "# X\n\nchanged\n")
    (root / "Project" / "y.md").unlink()
    write(root / "data.jsonl", '{"id": "1", "text": "one, changed"}\n')
    indexing.index(catalog)
    assert sorted(os.path.relpath(p, root).replace(os.sep, "/") for p in read) == ["Project/sub/x.md", "data.jsonl"]


@pytest.mark.parametrize("freed", [False, True])
def test_a_file_an_older_sift_left_out_for_its_path_is_read_again_once(corpus, home, monkeypatch, freed):
    # sift 1.11 left it out for good, its time recorded and nothing to say it waited: also once the path was free
    # again, as when the record went before sift was upgraded. Read again once, with every file without documents.
    catalog, root = corpus
    write(root / "data.jsonl", '{"id": "a.md", "text": "zebra in the record"}\n')
    write(root / "data" / "a.md", "# A\n\ngiraffe in the real file\n")
    indexing.index(catalog)
    assert Index(catalog).search("giraffe") == []  # the record keeps the path
    as_sift_1_11_left_it(catalog.store("notes"))
    write(root / "data.jsonl", '{"id": "b.md", "text": "another record"}\n')
    if freed:
        indexing.index(catalog)  # by sift 1.11: the path free, and nothing said of it
    catalog.close()
    fresh = Catalog(home)  # opening the store upgrades it
    indexing.index(fresh)
    assert [hit["id"] for hit in Index(fresh).search("giraffe")] == ["notes/data/a.md"]
    read, loaded = [], indexing._loaded
    monkeypatch.setattr(indexing, "_loaded", lambda path, fields: read.append(path) or loaded(path, fields))
    indexing.index(fresh)
    assert read == []  # once
    fresh.close()


def as_sift_1_11_left_it(db):
    """A store as sift 1.11 left it: nothing to say what waits for a path, and not yet upgraded."""
    db.execute("DROP TABLE IF EXISTS waiting")
    db.execute("DELETE FROM meta WHERE key = 'paths_waited'")
    db.commit()


def test_a_record_an_older_sift_left_out_of_a_file_that_kept_others_is_read_again_once(corpus, home):
    # Its file has documents, so it isn't among those without: but another file's lie below its name.
    catalog, root = corpus
    write(root / "data" / "a.md", "# A\n\ngiraffe in the real file\n")
    indexing.index(catalog)
    write(root / "data.jsonl", '{"id": "a.md", "text": "zebra in the record"}\n{"id": "b.md", "text": "another"}\n')
    write(root / "logs.jsonl", '{"id": "1", "text": "a log line"}\n')  # nothing below its name
    indexing.index(catalog)
    assert Index(catalog).search("zebra") == []  # the file keeps the path
    as_sift_1_11_left_it(catalog.store("notes"))
    catalog.close()
    fresh = Catalog(home)  # opening the store upgrades it
    marked = {p for (p,) in fresh.store("notes").execute("SELECT path FROM sources WHERE mtime IS NULL")}
    assert marked == {"data.jsonl"}
    (root / "data" / "a.md").unlink()
    indexing.index(fresh)
    assert [hit["id"] for hit in Index(fresh).search("zebra")] == ["notes/data/a.md"]
    fresh.close()


def test_an_attachment_an_older_sift_left_out_of_a_mail_that_kept_others_is_read_again_once(corpus, home):
    # The mail's own document lies beside its name, its attachments below it, as the real file does.
    catalog, root = corpus
    write(root / "offer" / "contract.md", "# Contract\n\ngiraffe in the real file\n")
    indexing.index(catalog)
    attached = [
        ("contract.md", b"# Contract\n\nzebra in the attachment\n", "text/markdown"),
        ("notes.md", b"# Notes\n\nmore\n", "text/markdown"),
    ]
    email_file(root, name="offer.eml", subject="Offer", body="see the attachments", attachments=attached)
    indexing.index(catalog)
    assert Index(catalog).search("zebra") == []  # the file keeps the path
    as_sift_1_11_left_it(catalog.store("notes"))
    catalog.close()
    fresh = Catalog(home)  # opening the store upgrades it
    assert {p for (p,) in fresh.store("notes").execute("SELECT path FROM sources WHERE mtime IS NULL")} == {"offer.eml"}
    fresh.close()


def test_a_store_sift_may_only_read_opens_unmarked(corpus, home):
    # sift 1.11 opened it; the marking, which writes, failed, and with it every search of the collection.
    catalog, _ = corpus
    as_sift_1_11_left_it(catalog.store("notes"))
    path = catalog.get("notes").store
    catalog.close()
    os.chmod(path, stat.S_IREAD)
    try:
        if os.access(path, os.W_OK):  # root may write it anyway: nothing here to try
            pytest.skip("this user may write a read-only file")
        fresh = Catalog(home)
        assert [hit["id"] for hit in Index(fresh).search("brush")] == ["notes/robots/brush.md"]
        assert store.get_meta(fresh.store("notes"), "paths_waited") is None  # marked once sift may write
        fresh.close()
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def test_a_document_the_index_refuses_for_another_reason_than_its_path_skips_its_file(corpus, monkeypatch, capsys):
    catalog, root = corpus
    write(root / "new.md", "# New\n")

    def refuse(*args):
        raise sqlite3.IntegrityError("CHECK constraint failed")

    monkeypatch.setattr(indexing, "_insert", refuse)
    indexing.index(catalog)
    assert "notes: new.md: skipped (new.md: new.md: the index refused the document)" in capsys.readouterr().err


def test_a_small_file_is_stored_with_no_statement_a_large_one_needs(corpus):
    # What a file of millions of records needs, its documents looked up by their paths, those that stay kept in a
    # table, those gone found by a query, and long documents' chunks looked for, cost each small file four statements
    # more: 20,000 notes took a tenth longer to index, and a fifth when only their times had changed.
    catalog, root = corpus
    db, asked = catalog.store("notes"), []
    write(root / "new.md", "# New\n\nA new note.\n")
    os.utime(root / "rl.md", ns=(1_600_000_000 * 10**9,) * 2)  # read again, as it is indexed
    db.set_trace_callback(asked.append)
    try:
        indexing.index(catalog)
    finally:
        db.set_trace_callback(None)
    assert sum(s.startswith("SELECT path, id, hash FROM docs WHERE source = ") for s in asked) == 2  # once a file
    assert sum(s.startswith("INSERT INTO fts(") for s in asked) == 1  # the new note's entries
    for costly in ("FROM docs WHERE path IN", "sift_staying", "FROM chunks WHERE id IN"):
        assert not [s for s in asked if costly in s], costly


def test_index_is_incremental(corpus, capsys):
    catalog, root = corpus
    ids = doc_ids(catalog)
    write(root / "rl.md", "# Reward shaping\n\nChanged body.\n")
    (root / "plain.txt").unlink()
    write(root / "new.md", "# New\n")
    os.utime(root / "robots" / "brush.md", ns=(10**18, 10**18))  # touched but unchanged
    indexing.index(catalog)
    after = doc_ids(catalog)
    assert set(after) == {"rl.md", "robots/brush.md", "new.md"}
    assert after["robots/brush.md"] == ids["robots/brush.md"]
    assert after["rl.md"] != ids["rl.md"]
    assert catalog.store("notes").execute("SELECT mtime FROM sources WHERE path = 'robots/brush.md'").fetchone() == (
        10**18,
    )
    assert "1 added, 1 changed, 1 removed" in capsys.readouterr().err
    orphans = (
        catalog.store("notes")
        .execute("SELECT count(*) FROM chunks WHERE doc NOT IN (SELECT id FROM docs)")
        .fetchone()[0]
    )
    assert orphans == 0


@pytest.mark.usefixtures("terminal")
def test_index_says_what_it_scans_and_counts_file_by_file(notes, home, monkeypatch, capsys):
    # A batch of large PDFs or a slow cloud folder took minutes before the first line; sift looked hung.
    catalog = Catalog(home)
    catalog.add(notes, "notes")
    walk, said = indexing.walk, []
    monkeypatch.setattr(indexing, "walk", lambda root, *more: said.append(capsys.readouterr().err) or walk(root, *more))
    indexing.index(catalog)
    assert said[0].startswith(f"notes: scanning {notes}")  # before walking the folder
    err = capsys.readouterr().err
    assert all("\rnotes: [" in err and f" {n}/3 files" in err for n in (1, 2, 3))  # one batch, file by file


def test_index_counts_many_files_in_steps(notes, home, monkeypatch, capsys):
    monkeypatch.setattr(indexing, "PROGRESS_LINES", 1)  # as with 200 lines for many thousand files
    catalog = Catalog(home)
    catalog.add(notes, "notes")
    indexing.index(catalog)
    err = capsys.readouterr().err
    assert " 3/3 files" in err
    assert " 1/3 files" not in err


def test_index_in_several_batches(notes, home, monkeypatch):
    monkeypatch.setattr(indexing, "WRITE_BATCH", 2)
    catalog = Catalog(home)
    catalog.add(notes, "notes")
    indexing.index(catalog)
    assert len(doc_ids(catalog)) == 3


def test_index_parses_in_worker_processes(notes, tmp_path, monkeypatch, pools):
    serial = Catalog(tmp_path / "serial")
    serial.add(notes, "notes")
    indexing.index(serial)
    assert pools == []

    monkeypatch.setattr(indexing, "cpus", lambda: 3)  # two workers, one CPU for writing
    monkeypatch.setattr(indexing, "FILES_PER_WORKER", 1)
    monkeypatch.setattr(indexing, "WRITE_BATCH", 2)  # several batches in flight
    parallel = Catalog(tmp_path / "parallel")
    parallel.add(notes, "notes")
    indexing.index(parallel)
    # spawned, not forked, on every platform: forking a process with threads (PyTorch starts some) can deadlock
    assert pools == [(2, "spawn")]
    assert index_contents(parallel.store("notes")) == index_contents(serial.store("notes"))


def sized(*paths):
    """Files as the worker processes are handed them: (path, size)."""
    return [(str(path), path.stat().st_size) for path in paths]


def no_process_left():
    """Whether every process this one started has ended, waited for at most ten seconds."""
    deadline = time.monotonic() + 10
    while multiprocessing.active_children() and time.monotonic() < deadline:
        time.sleep(0.01)
    return not multiprocessing.active_children()


@pytest.fixture
def three_files(tmp_path):
    """Three notes, of which the first ends or hangs a reader from tests/workers.py: (poison, hangs, fine)."""
    files = [tmp_path / name for name in ("poison.md", "hangs.md", "fine.md")]
    for file in files:
        write(file, f"# {file.stem}\n")
    return files


def test_a_file_that_ends_its_reader_is_left_out_and_the_others_are_read(notes, home, monkeypatch, capsys):
    # A reader's native code that crashed took its worker process along. The pool said no more than that it was
    # broken: `sift index` ended in a traceback, and every later run met the same file in the same batch.
    write(notes / "poison.md", "# Poison\n")
    monkeypatch.setattr(indexing, "cpus", lambda: 3)  # two workers, one CPU for writing
    monkeypatch.setattr(indexing, "FILES_PER_WORKER", 1)
    monkeypatch.setattr(indexing, "WRITE_BATCH", 2)  # several batches in flight, as one of them ends the pool
    monkeypatch.setattr(indexing, "_loaded", workers.ending)
    catalog = Catalog(home)
    catalog.add(notes, "notes")
    indexing.index(catalog)
    assert set(doc_ids(catalog)) == {"plain.txt", "rl.md", "robots/brush.md"}
    said = capsys.readouterr().err
    assert "notes: poison.md: skipped (Stopped: its reader ended the process it ran in)" in said
    assert "notes: 4 files, 3 added, 0 changed, 0 removed, 1 unreadable" in said
    assert no_process_left()


def test_the_files_started_with_one_that_ends_the_pool_are_read_alone(three_files, monkeypatch):
    # The pool tells no more than that it broke: which of the files not read yet did it, each alone tells.
    poison, _, fine = three_files
    monkeypatch.setattr(indexing, "_loaded", workers.ending)
    with indexing._Workers(2, {}) as reading:
        (stopped, (docs, problems)) = reading.results(reading.start(sized(poison, fine)))
    assert isinstance(stopped, indexing.Stopped)
    assert str(stopped) == "its reader ended the process it ran in"
    assert ([doc.title for doc in docs], problems) == (["fine"], [])
    assert no_process_left()


def test_a_file_whose_reader_comes_to_no_end_is_stopped(three_files, monkeypatch):
    # A reader in a loop held `sift index` for good: no wait had an end. Its process is ended, at once, and the file
    # left out; a file gets its time by its size, and here the one that hangs gets none.
    _, hangs, fine = three_files
    monkeypatch.setattr(indexing, "_loaded", workers.hanging)
    monkeypatch.setattr(indexing, "_patience", lambda files: 0 if any("hangs" in path for path, _ in files) else 600)
    with indexing._Workers(2, {}) as reading:
        (stopped, (docs, _)) = reading.results(reading.start(sized(hangs, fine)))
    assert isinstance(stopped, indexing.Stopped)
    assert str(stopped) == "its reader came to no end in 0:00:00"
    assert [doc.title for doc in docs] == ["fine"]
    assert no_process_left()


def test_a_batch_started_in_a_pool_another_batch_ended_starts_again(three_files, monkeypatch):
    # The next batch is started while the one before is stored: in a pool the one before then ends. One process, so
    # that the batch after the poison is never read in it.
    poison, _, fine = three_files
    monkeypatch.setattr(indexing, "_loaded", workers.ending)
    with indexing._Workers(1, {}) as reading:
        first, second = reading.start(sized(poison)), reading.start(sized(fine))
        assert isinstance(reading.results(first)[0], indexing.Stopped)
        ((docs, _),) = reading.results(second)
    assert [doc.title for doc in docs] == ["fine"]
    assert no_process_left()


def test_stopping_sift_ends_the_worker_processes_at_once(three_files, monkeypatch):
    # Ctrl+C waited for every file started, the next batch's too: thousands of them.
    _, hangs, _ = three_files
    monkeypatch.setattr(indexing, "_loaded", workers.hanging)

    def stopped():
        with indexing._Workers(1, {}) as reading:
            reading.start(sized(hangs))
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        stopped()
    assert no_process_left()


def test_a_file_gets_more_time_the_larger_it_is():
    assert indexing._patience([("a.pdf", 0)]) == indexing.READ_SECONDS
    assert indexing._patience([("a.pdf", 50 * indexing.READ_RATE), ("b.pdf", 50 * indexing.READ_RATE)]) == (
        indexing.READ_SECONDS + 100
    )


def test_small_updates_skip_worker_processes(corpus, monkeypatch, pools):
    catalog, root = corpus
    monkeypatch.setattr(indexing, "cpus", lambda: 64)
    write(root / "new.md", "# New\n")
    indexing.index(catalog)
    assert pools == []
    assert "new.md" in doc_ids(catalog)


@pytest.mark.parametrize(
    "cpus, files, expected",
    [
        (64, 0, 0),
        (64, 999, 0),  # too few files to be worth a process
        (64, 5_000, 5),  # one worker per FILES_PER_WORKER files
        (4, 1_000_000, 3),  # one CPU stays free for the process writing to SQLite
        (1, 1_000_000, 0),
    ],
)
def test_worker_count_adapts_to_machine_and_workload(monkeypatch, cpus, files, expected):
    monkeypatch.setattr(indexing, "cpus", lambda: cpus)
    assert indexing.workers(files) == expected


def test_cpus_respects_what_the_platform_allows(monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 6, raising=False)  # Python 3.13+
    assert indexing.cpus() == 6
    monkeypatch.delattr(os, "process_cpu_count")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)  # Linux
    assert indexing.cpus() == 3
    monkeypatch.delattr(os, "sched_getaffinity")
    monkeypatch.setattr(os, "cpu_count", lambda: None)  # unknown
    assert indexing.cpus() == 1


def test_index_selected_collections(corpus, tmp_path):
    catalog, _ = corpus
    other = tmp_path / "other"
    write(other / "a.md", "# A\n")
    catalog.add(other, "other")
    indexing.index(catalog, ["notes"])
    assert "a.md" not in doc_ids(catalog)
    with pytest.raises(LookupError, match="no collection named 'nope'; the collections are notes, other"):
        indexing.index(catalog, ["other", "nope"])
    assert "a.md" not in doc_ids(catalog)  # nothing indexed before the error


def test_index_skips_missing_folder_without_deleting(corpus, capsys):
    catalog, root = corpus
    root.rename(root.with_name("moved"))
    indexing.index(catalog)
    assert len(doc_ids(catalog)) == 3
    assert "folder not found" in capsys.readouterr().err


def test_a_folder_that_cannot_be_listed_keeps_its_documents(corpus, monkeypatch, capsys):
    # A share that briefly denied access looked empty: sift dropped its documents, and the vectors embedded for them.
    catalog, root = corpus
    real = os.scandir

    def scandir(path):
        if os.path.basename(path) == "robots":
            raise PermissionError(13, "Access is denied")
        return real(path)

    monkeypatch.setattr(indexing.os, "scandir", scandir)
    indexing.index(catalog)
    assert "robots/brush.md" in doc_ids(catalog)
    assert "notes: robots/ can't be read (Access is denied): 1 file in it kept as indexed" in capsys.readouterr().err
    monkeypatch.setattr(indexing.os, "scandir", real)  # not undo(): that would undo the fixtures' too, the home too
    shutil.rmtree(root / "robots")  # gone for good: its documents go
    indexing.index(catalog)
    assert "robots/brush.md" not in doc_ids(catalog)


def test_a_folder_left_without_any_file_keeps_its_documents(corpus, capsys):
    # A drive not mounted leaves its folder there, empty, and so does a folder being filled anew: `sift index` took
    # every file for deleted, and dropped the documents with the vectors embedded for them, days of a GPU's work.
    catalog, root = corpus
    held = doc_ids(catalog)
    shutil.rmtree(root)
    root.mkdir()
    capsys.readouterr()
    indexing.index(catalog)
    assert doc_ids(catalog) == held
    said = capsys.readouterr().err
    assert f"notes: none of its 3 files is left in {root}: kept as indexed" in said
    assert "`sift rm notes` forgets them" in said
    write(root / "rl.md", "# Reward shaping\n\nWritten anew.\n")  # one is there again: the others are gone for good
    indexing.index(catalog)
    assert set(doc_ids(catalog)) == {"rl.md"}


class Listing(list):
    """What os.scandir gives, made of other entries: a context manager that iterates them, and may break off."""

    def __init__(self, entries, broken=None):
        super().__init__(entries)
        self.broken = broken

    def __iter__(self):
        yield from super().__iter__()
        if self.broken:
            raise self.broken

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Denied:
    """A directory entry that can be listed but not stat'ed."""

    def __init__(self, entry):
        self.name, self.path, self.entry = entry.name, entry.path, entry

    def is_dir(self, follow_symlinks):
        return self.entry.is_dir(follow_symlinks=follow_symlinks)

    def stat(self):
        raise PermissionError(13, "Access is denied")


def test_a_file_that_cannot_be_stated_keeps_its_documents(corpus, monkeypatch, capsys):
    # As a folder that can't be listed: a file a share denies for the moment is no file deleted.
    catalog, root = corpus
    real = os.scandir

    def scandir(path):
        with real(path) as entries:
            return Listing([Denied(e) if e.name in ("rl.md", "new.md") else e for e in entries])

    write(root / "new.md", "# New\n")
    monkeypatch.setattr(indexing.os, "scandir", scandir)
    indexing.index(catalog)
    assert "rl.md" in doc_ids(catalog)
    err = capsys.readouterr().err
    assert "notes: rl.md can't be read (Access is denied): kept as indexed" in err
    assert "notes: new.md can't be read (Access is denied): left for the next index" in err


def test_a_listing_that_breaks_off_keeps_the_folder_s_documents(corpus, monkeypatch, capsys):
    catalog, _ = corpus
    real = os.scandir

    def scandir(path):
        if os.path.basename(path) == "robots":  # the share drops before its first entry
            return Listing([], OSError(5, "Input/output error"))
        return real(path)

    monkeypatch.setattr(indexing.os, "scandir", scandir)
    indexing.index(catalog)
    assert "robots/brush.md" in doc_ids(catalog)
    assert "notes: robots/ can't be read (Input/output error): 1 file in it kept as indexed" in capsys.readouterr().err


def test_a_folder_that_became_a_file_is_gone(tmp_path, monkeypatch):
    def scandir(path):
        raise NotADirectoryError(20, "Not a directory")

    monkeypatch.setattr(indexing.os, "scandir", scandir)
    unlisted = []
    assert list(indexing.walk(str(tmp_path), None, unlisted)) == []
    assert unlisted == []


def test_reindex_keeps_a_file_it_cannot_stat_and_says_so(corpus, monkeypatch):
    catalog, _ = corpus
    real = os.stat

    def stat(path, *args, **kwargs):
        if str(path).endswith("rl.md"):
            raise PermissionError(13, "Access is denied")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(indexing.os, "stat", stat)
    ((entry, _),) = indexing.reindex(catalog, ["notes/rl.md"]).files
    assert entry["status"] == "unreadable"  # not "removed"
    assert "rl.md" in doc_ids(catalog)


def test_walk_names_the_folders_it_cannot_list_when_asked(tmp_path, monkeypatch):
    def scandir(path):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(indexing.os, "scandir", scandir)
    assert list(indexing.walk(str(tmp_path))) == []
    unlisted = []
    assert list(indexing.walk(str(tmp_path), None, unlisted)) == []
    assert [(rel, error.strerror) for rel, error in unlisted] == [("", "Access is denied")]


def test_a_collection_whose_folder_cannot_be_listed_keeps_everything(corpus, monkeypatch, capsys):
    catalog, _ = corpus

    def scandir(path):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(indexing.os, "scandir", scandir)
    indexing.index(catalog)
    assert len(doc_ids(catalog)) == 3
    assert (
        "notes: its folder can't be read (Access is denied): 3 files in it kept as indexed" in capsys.readouterr().err
    )


def test_index_skips_unreadable_files_and_retries_later(notes, home, monkeypatch, capsys):
    real_load = readers.load
    monkeypatch.setattr(readers, "load", lambda path, fields=None: None if path.endswith("rl.md") else real_load(path))
    catalog = Catalog(home)
    catalog.add(notes, "notes")
    indexing.index(catalog)
    assert "rl.md" not in doc_ids(catalog)
    assert "1 unreadable" in capsys.readouterr().err
    monkeypatch.setattr(readers, "load", real_load)
    indexing.index(catalog)
    assert "rl.md" in doc_ids(catalog)


def test_index_skips_file_names_sqlite_cannot_store(corpus, monkeypatch, capsys):
    # Linux file names may hold bytes that aren't UTF-8, which Python keeps as surrogates and SQLite refuses:
    # one such file aborted the whole run.
    catalog, root = corpus
    real_walk = indexing.walk

    def walk(path, *more):
        yield os.path.join(path, "r\udce9sum\udce9.md"), 1, 1
        yield from real_walk(path, *more)

    monkeypatch.setattr(indexing, "walk", walk)
    write(root / "new.md", "# New\n")
    indexing.index(catalog)
    assert "new.md" in doc_ids(catalog)
    assert "unsupported file name, skipped: " in capsys.readouterr().err


# --- titles from file names --------------------------------------------------


def titles(catalog):
    return dict(catalog.store("notes").execute("SELECT path, title FROM docs"))


def test_a_collection_retitled_from_the_text_the_index_holds(corpus, monkeypatch):
    catalog, root = corpus
    write(root / "Hans Müller.md", "## Summary\n\nKey account at Acme.\n")
    write(root / "records.jsonl", '{"id": "r1", "text": "## Summary\\n\\nA record."}\n')
    indexing.index(catalog)
    monkeypatch.setattr(readers, "load", lambda *args: pytest.fail("a file was read again"))
    assert indexing.retitle(catalog, "notes", "heading") == (0, 0)  # as they are
    assert indexing.retitle(catalog, "notes", "name") == (2, 0)
    assert titles(catalog) == {
        "Hans Müller.md": "Hans Müller",
        "rl.md": "rl",
        "robots/brush.md": "Brush stroke planning",  # its title field wins
        "plain.txt": "plain",
        "records/r1": "Summary",  # a record's title comes from its fields
    }
    assert keywords_agree(catalog.store("notes"))
    for query, found in (("title: Müller", ["notes/Hans Müller.md"]), ("title: Summary", ["notes/records/r1"])):
        assert [h["id"] for h in Index(catalog).search(query)] == found
    assert indexing.retitle(catalog, "notes", "heading") == (2, 0)
    assert titles(catalog)["Hans Müller.md"] == "Summary"
    assert keywords_agree(catalog.store("notes"))


def test_retitled_notes_stay_retitled_when_indexed_again(corpus):
    catalog, root = corpus
    indexing.retitle(catalog, "notes", "name")
    write(root / "Hans Müller.md", "## Summary\n")
    indexing.index(catalog)
    assert titles(catalog)["Hans Müller.md"] == "Hans Müller"
    assert titles(catalog)["rl.md"] == "rl"


def test_retitling_drops_vectors_embedded_with_the_old_title(embedded):
    # Passages after a document's first are embedded with its title before them: "Summary" would stay in them.
    catalog, root = embedded
    write(root / "long.md", "## Summary\n\n" + "first part " * 200 + "\n\n## Details\n\n" + "second part " * 200)
    indexing.index(catalog)
    embed.embed_pending(catalog)
    before = store.get_meta(catalog.store("notes"), "version")
    assert indexing.retitle(catalog, "notes", "name") == (2, 1)
    missing = (
        catalog.store("notes")
        .execute(
            "SELECT d.path, c.seq FROM chunks c JOIN docs d ON d.id = c.doc "
            "WHERE c.id NOT IN (SELECT chunk FROM vectors)"
        )
        .fetchall()
    )
    assert missing == [("long.md", 1)]  # the first passage is embedded without its title
    assert store.get_meta(catalog.store("notes"), "version") != before  # a running server loads the codes again


def test_retitling_an_unknown_collection(corpus):
    with pytest.raises(LookupError, match="no collection named 'nope'; the collections are notes"):
        indexing.retitle(corpus[0], "nope", "name")


def test_a_few_big_files_still_get_workers(monkeypatch):
    # 424 JSON Lines files of arXiv are few files but gigabytes of records to parse
    monkeypatch.setattr(indexing, "cpus", lambda: 16)
    assert indexing.workers(424, 0) == 0
    assert indexing.workers(424, 5 << 30) == 15
    assert indexing.workers(2, 3 * indexing.BYTES_PER_WORKER) == 3


def test_batches_hold_at_most_so_many_files_and_bytes(monkeypatch):
    monkeypatch.setattr(indexing, "WRITE_BATCH", 3)
    monkeypatch.setattr(indexing, "BATCH_BYTES", 100)
    todo = [(f"f{i}", f"f{i}", 0, size) for i, size in enumerate([10, 10, 10, 10, 95, 200, 1])]
    assert [[t[0] for t in b] for b in indexing._batches(todo)] == [["f0", "f1", "f2"], ["f3"], ["f4"], ["f5"], ["f6"]]
    assert indexing._batches([]) == []


MAIL = "From: a@example.com\nSubject: {subject}\nDate: {date}\nContent-Type: text/plain; charset={charset}\n\n{body}\n"


@pytest.mark.parametrize(
    "date, charset, body",
    [
        ("Fri, 31 Dec 9999 23:00:00 -0500", "utf-8", "wombat"),  # past 9999 in UTC: the date stays as written
        ("Fri, 1 Mar 2024 10:00:00 +0100", "utf-7", "wombat +2D0- half an emoji"),  # UTF-7 can encode half of one
    ],
)
def test_a_mail_python_reads_oddly_is_indexed(corpus, date, charset, body):
    # Both were skipped, the second on every run: a date Python couldn't move to UTC, a half character SQLite can't
    # store.
    catalog, root = corpus
    write(root / "odd.eml", MAIL.format(subject="Odd", date=date, charset=charset, body=body))
    indexing.index(catalog)
    assert [hit["id"] for hit in Index(catalog).search("wombat")] == ["notes/odd.eml"]


def test_a_text_file_larger_than_a_document_may_be_is_left_out_unread(corpus, monkeypatch, capsys):
    # SQLite stores a billion bytes as one value at most. A larger text file was read whole, cut into passages and
    # handed over by its worker process, gigabytes each time, then refused by the store: on every run, as a file
    # refused is read again by the next.
    catalog, root = corpus
    monkeypatch.setattr(readers, "MOST_TEXT", 50)
    write(root / "huge.txt", "x" * 51)
    write(root / "full.txt", "x" * 50)
    indexing.index(catalog)
    assert {"full.txt", "rl.md"} <= set(doc_ids(catalog))
    assert "huge.txt" not in doc_ids(catalog)
    said = capsys.readouterr().err
    assert "notes: huge.txt: skipped (51 bytes: a document holds at most 50 bytes of text; split the file)" in said
    read, real = [], readers.load
    monkeypatch.setattr(readers, "load", lambda *args: read.append(args) or real(*args))
    indexing.index(catalog)
    assert read == []  # known by its time and size: left alone until it changes


class Pipe(Denied):
    """A directory entry that is a named pipe: no file with an end to read to."""

    def stat(self):
        return os.stat_result((stat.S_IFIFO | 0o644, *tuple(self.entry.stat())[1:]))


def test_walk_passes_over_what_is_no_file(notes, monkeypatch):
    # A named pipe called x.md was opened as a file is, and its reader waited for an end that never came; a folder
    # linked under such a name was "unreadable" on every run.
    real = os.scandir

    def scandir(path):
        with real(path) as entries:
            return Listing([Pipe(e) if e.name == "rl.md" else e for e in entries])

    monkeypatch.setattr(indexing.os, "scandir", scandir)
    found = sorted(os.path.relpath(p, notes).replace(os.sep, "/") for p, _, _ in indexing.walk(str(notes)))
    assert found == ["plain.txt", "robots/brush.md"]
