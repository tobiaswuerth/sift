"""`sift index`: scanning folders, reading new and changed files in worker processes, and keeping the index in
step with the files, incrementally."""

import os
import shutil

import pytest

from sift import embed, indexing, readers, store
from sift.ignore import Ignore
from sift.search import Index

from .conftest import index_contents, keywords_agree, write


def doc_ids(db):
    return dict(db.execute("SELECT path, id FROM docs"))


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
    db, root = corpus
    write(root / ".md", "# Nameless\n")
    write(root / ".notes.md", "# A note whose name starts with a dot\n")
    found = {os.path.relpath(p, root) for p, _, _ in indexing.walk(str(root))}
    assert ".notes.md" in found
    assert ".md" not in found
    indexing.index(db)
    assert ".notes.md" in doc_ids(db)


def test_walk_leaves_out_what_siftignore_names(notes):
    write(notes / ".siftignore", "plain.txt\nrobots/\n")
    write(notes / "robots" / "deep" / "x.md", "# Inside a folder left out\n")
    found = [os.path.relpath(p, notes) for p, _, _ in indexing.walk(str(notes), Ignore.of(notes))]
    assert found == ["rl.md"]


def test_index_follows_siftignore_as_it_changes(corpus, capsys):
    # An app keeps its own files out, a user drafts or exports; and what is left out after all leaves the index.
    db, root = corpus
    write(root / ".siftignore", "# drafts and exports\nrobots/\n*.txt\n")
    indexing.index(db)
    assert sorted(doc_ids(db)) == ["rl.md"]
    assert "2 removed" in capsys.readouterr().err
    write(root / ".siftignore", "robots/\n")
    indexing.index(db)
    assert sorted(doc_ids(db)) == ["plain.txt", "rl.md"]


def test_a_collection_whose_siftignore_cannot_be_read_is_skipped(corpus, tmp_path, capsys):
    # Indexing all of it would take in the very files the .siftignore keeps out, as an offline OneDrive file can't
    # be read; its documents stay, and the collections after it are indexed still.
    db, root = corpus
    write(tmp_path / "other" / "x.md", "# X\n")
    store.add_collection(db, tmp_path / "other", "other")
    (root / ".siftignore").mkdir()
    before = doc_ids(db)
    indexing.index(db)
    assert "notes: skipped, as its .siftignore can't be read: " in capsys.readouterr().err
    assert doc_ids(db).keys() == before.keys() | {"x.md"}


# --- indexing ----------------------------------------------------------------


def test_index_picks_text_and_markdown_only(corpus):
    assert sorted(doc_ids(corpus[0])) == ["plain.txt", "rl.md", "robots/brush.md"]


def test_index_is_incremental(corpus, capsys):
    db, root = corpus
    ids = doc_ids(db)
    write(root / "rl.md", "# Reward shaping\n\nChanged body.\n")
    (root / "plain.txt").unlink()
    write(root / "new.md", "# New\n")
    os.utime(root / "robots" / "brush.md", ns=(10**18, 10**18))  # touched but unchanged
    indexing.index(db)
    after = doc_ids(db)
    assert set(after) == {"rl.md", "robots/brush.md", "new.md"}
    assert after["robots/brush.md"] == ids["robots/brush.md"]
    assert after["rl.md"] != ids["rl.md"]
    assert db.execute("SELECT mtime FROM sources WHERE path = 'robots/brush.md'").fetchone() == (10**18,)
    assert "1 added, 1 changed, 1 removed" in capsys.readouterr().err
    orphans = db.execute("SELECT count(*) FROM chunks WHERE doc NOT IN (SELECT id FROM docs)").fetchone()[0]
    assert orphans == 0


def test_index_says_what_it_scans_and_counts_file_by_file(notes, home, monkeypatch, capsys):
    # A batch of large PDFs or a slow cloud folder took minutes before the first line; sift looked hung.
    db = store.connect(home)
    store.add_collection(db, notes, "notes")
    walk, said = indexing.walk, []
    monkeypatch.setattr(indexing, "walk", lambda root, *more: said.append(capsys.readouterr().err) or walk(root, *more))
    indexing.index(db)
    assert said[0].startswith(f"notes: scanning {notes}")  # before walking the folder
    err = capsys.readouterr().err
    assert all("\rnotes: [" in err and f" {n}/3 files" in err for n in (1, 2, 3))  # one batch, file by file


def test_index_counts_many_files_in_steps(notes, home, monkeypatch, capsys):
    monkeypatch.setattr(indexing, "PROGRESS_LINES", 1)  # as with 200 lines for many thousand files
    db = store.connect(home)
    store.add_collection(db, notes, "notes")
    indexing.index(db)
    err = capsys.readouterr().err
    assert " 3/3 files" in err
    assert " 1/3 files" not in err


def test_index_in_several_batches(notes, home, monkeypatch):
    monkeypatch.setattr(indexing, "WRITE_BATCH", 2)
    db = store.connect(home)
    store.add_collection(db, notes, "notes")
    indexing.index(db)
    assert len(doc_ids(db)) == 3


def test_index_parses_in_worker_processes(notes, tmp_path, monkeypatch, pools):
    serial = store.connect(tmp_path / "serial")
    store.add_collection(serial, notes, "notes")
    indexing.index(serial)
    assert pools == []

    monkeypatch.setattr(indexing, "_cpus", lambda: 3)  # two workers, one CPU for writing
    monkeypatch.setattr(indexing, "FILES_PER_WORKER", 1)
    monkeypatch.setattr(indexing, "WRITE_BATCH", 2)  # several batches in flight
    parallel = store.connect(tmp_path / "parallel")
    store.add_collection(parallel, notes, "notes")
    indexing.index(parallel)
    # spawned, not forked, on every platform: forking a process with threads (PyTorch starts some) can deadlock
    assert pools == [(2, "spawn")]
    assert index_contents(parallel) == index_contents(serial)


def test_small_updates_skip_worker_processes(corpus, monkeypatch, pools):
    db, root = corpus
    monkeypatch.setattr(indexing, "_cpus", lambda: 64)
    write(root / "new.md", "# New\n")
    indexing.index(db)
    assert pools == []
    assert "new.md" in doc_ids(db)


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
    monkeypatch.setattr(indexing, "_cpus", lambda: cpus)
    assert indexing.workers(files) == expected


def test_cpus_respects_what_the_platform_allows(monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 6, raising=False)  # Python 3.13+
    assert indexing._cpus() == 6
    monkeypatch.delattr(os, "process_cpu_count")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)  # Linux
    assert indexing._cpus() == 3
    monkeypatch.delattr(os, "sched_getaffinity")
    monkeypatch.setattr(os, "cpu_count", lambda: None)  # unknown
    assert indexing._cpus() == 1


def test_index_selected_collections(corpus, tmp_path):
    db, _ = corpus
    other = tmp_path / "other"
    write(other / "a.md", "# A\n")
    store.add_collection(db, other, "other")
    indexing.index(db, ["notes"])
    assert "a.md" not in doc_ids(db)
    with pytest.raises(SystemExit, match="'nope'"):
        indexing.index(db, ["nope"])


def test_index_skips_missing_folder_without_deleting(corpus, capsys):
    db, root = corpus
    root.rename(root.with_name("moved"))
    indexing.index(db)
    assert len(doc_ids(db)) == 3
    assert "folder not found" in capsys.readouterr().err


def test_a_folder_that_cannot_be_listed_keeps_its_documents(corpus, monkeypatch, capsys):
    # A share that briefly denied access looked empty: sift dropped its documents, and the vectors embedded for them.
    db, root = corpus
    real = os.scandir

    def scandir(path):
        if os.path.basename(path) == "robots":
            raise PermissionError(13, "Access is denied")
        return real(path)

    monkeypatch.setattr(indexing.os, "scandir", scandir)
    indexing.index(db)
    assert "robots/brush.md" in doc_ids(db)
    assert "notes: robots/ can't be read (Access is denied): 1 file in it kept as indexed" in capsys.readouterr().err
    monkeypatch.undo()
    shutil.rmtree(root / "robots")  # gone for good: its documents go
    indexing.index(db)
    assert "robots/brush.md" not in doc_ids(db)


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
    db, root = corpus
    real = os.scandir

    def scandir(path):
        with real(path) as entries:
            return Listing([Denied(e) if e.name in ("rl.md", "new.md") else e for e in entries])

    write(root / "new.md", "# New\n")
    monkeypatch.setattr(indexing.os, "scandir", scandir)
    indexing.index(db)
    assert "rl.md" in doc_ids(db)
    err = capsys.readouterr().err
    assert "notes: rl.md can't be read (Access is denied): kept as indexed" in err
    assert "notes: new.md can't be read (Access is denied): left for the next index" in err


def test_a_listing_that_breaks_off_keeps_the_folder_s_documents(corpus, monkeypatch, capsys):
    db, _ = corpus
    real = os.scandir

    def scandir(path):
        if os.path.basename(path) == "robots":  # the share drops before its first entry
            return Listing([], OSError(5, "Input/output error"))
        return real(path)

    monkeypatch.setattr(indexing.os, "scandir", scandir)
    indexing.index(db)
    assert "robots/brush.md" in doc_ids(db)
    assert "notes: robots/ can't be read (Input/output error): 1 file in it kept as indexed" in capsys.readouterr().err


def test_a_folder_that_became_a_file_is_gone(tmp_path, monkeypatch):
    def scandir(path):
        raise NotADirectoryError(20, "Not a directory")

    monkeypatch.setattr(indexing.os, "scandir", scandir)
    unlisted = []
    assert list(indexing.walk(str(tmp_path), None, unlisted)) == []
    assert unlisted == []


def test_reindex_keeps_a_file_it_cannot_stat_and_says_so(corpus, monkeypatch):
    db, _ = corpus
    real = os.stat

    def stat(path, *args, **kwargs):
        if str(path).endswith("rl.md"):
            raise PermissionError(13, "Access is denied")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(indexing.os, "stat", stat)
    ((entry, _),) = indexing.reindex(db, ["notes/rl.md"]).files
    assert entry["status"] == "unreadable"  # not "removed"
    assert "rl.md" in doc_ids(db)


def test_walk_names_the_folders_it_cannot_list_when_asked(tmp_path, monkeypatch):
    def scandir(path):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(indexing.os, "scandir", scandir)
    assert list(indexing.walk(str(tmp_path))) == []
    unlisted = []
    assert list(indexing.walk(str(tmp_path), None, unlisted)) == []
    assert [(rel, error.strerror) for rel, error in unlisted] == [("", "Access is denied")]


def test_a_collection_whose_folder_cannot_be_listed_keeps_everything(corpus, monkeypatch, capsys):
    db, _ = corpus

    def scandir(path):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(indexing.os, "scandir", scandir)
    indexing.index(db)
    assert len(doc_ids(db)) == 3
    assert (
        "notes: its folder can't be read (Access is denied): 3 files in it kept as indexed" in capsys.readouterr().err
    )


def test_index_skips_unreadable_files_and_retries_later(notes, home, monkeypatch, capsys):
    real_load = readers.load
    monkeypatch.setattr(readers, "load", lambda path, fields=None: None if path.endswith("rl.md") else real_load(path))
    db = store.connect(home)
    store.add_collection(db, notes, "notes")
    indexing.index(db)
    assert "rl.md" not in doc_ids(db)
    assert "1 unreadable" in capsys.readouterr().err
    monkeypatch.setattr(readers, "load", real_load)
    indexing.index(db)
    assert "rl.md" in doc_ids(db)


def test_index_skips_file_names_sqlite_cannot_store(corpus, monkeypatch, capsys):
    # Linux file names may hold bytes that aren't UTF-8, which Python keeps as surrogates and SQLite refuses:
    # one such file aborted the whole run.
    db, root = corpus
    real_walk = indexing.walk

    def walk(path, *more):
        yield os.path.join(path, "r\udce9sum\udce9.md"), 1, 1
        yield from real_walk(path, *more)

    monkeypatch.setattr(indexing, "walk", walk)
    write(root / "new.md", "# New\n")
    indexing.index(db)
    assert "new.md" in doc_ids(db)
    assert "unsupported file name, skipped: " in capsys.readouterr().err


# --- titles from file names --------------------------------------------------


def titles(db):
    return dict(db.execute("SELECT path, title FROM docs"))


def test_a_collection_retitled_from_the_text_the_index_holds(corpus, monkeypatch):
    db, root = corpus
    write(root / "Hans Müller.md", "## Summary\n\nKey account at Acme.\n")
    write(root / "records.jsonl", '{"id": "r1", "text": "## Summary\\n\\nA record."}\n')
    indexing.index(db)
    monkeypatch.setattr(readers, "load", lambda *args: pytest.fail("a file was read again"))
    assert indexing.retitle(db, "notes", "heading") == (0, 0)  # as they are
    assert indexing.retitle(db, "notes", "name") == (2, 0)
    assert titles(db) == {
        "Hans Müller.md": "Hans Müller",
        "rl.md": "rl",
        "robots/brush.md": "Brush stroke planning",  # its title field wins
        "plain.txt": "plain",
        "records/r1": "Summary",  # a record's title comes from its fields
    }
    assert keywords_agree(db)
    for query, found in (("title: Müller", ["notes/Hans Müller.md"]), ("title: Summary", ["notes/records/r1"])):
        assert [h["id"] for h in Index(db).search(query)] == found
    assert indexing.retitle(db, "notes", "heading") == (2, 0)
    assert titles(db)["Hans Müller.md"] == "Summary"
    assert keywords_agree(db)


def test_retitled_notes_stay_retitled_when_indexed_again(corpus):
    db, root = corpus
    indexing.retitle(db, "notes", "name")
    write(root / "Hans Müller.md", "## Summary\n")
    indexing.index(db)
    assert titles(db)["Hans Müller.md"] == "Hans Müller"
    assert titles(db)["rl.md"] == "rl"


def test_retitling_drops_vectors_embedded_with_the_old_title(embedded):
    # Passages after a document's first are embedded with its title before them: "Summary" would stay in them.
    db, root = embedded
    write(root / "long.md", "## Summary\n\n" + "first part " * 200 + "\n\n## Details\n\n" + "second part " * 200)
    indexing.index(db)
    embed.embed_pending(db)
    before = store.get_meta(db, "version")
    assert indexing.retitle(db, "notes", "name") == (2, 1)
    missing = db.execute(
        "SELECT d.path, c.seq FROM chunks c JOIN docs d ON d.id = c.doc WHERE c.id NOT IN (SELECT chunk FROM vectors)"
    ).fetchall()
    assert missing == [("long.md", 1)]  # the first passage is embedded without its title
    assert store.get_meta(db, "version") != before  # a running server loads the vectors again


def test_retitling_an_unknown_collection(corpus):
    with pytest.raises(SystemExit, match="no collection named 'nope'"):
        indexing.retitle(corpus[0], "nope", "name")


def test_a_few_big_files_still_get_workers(monkeypatch):
    # 424 JSON Lines files of arXiv are few files but gigabytes of records to parse
    monkeypatch.setattr(indexing, "_cpus", lambda: 16)
    assert indexing.workers(424, 0) == 0
    assert indexing.workers(424, 5 << 30) == 15
    assert indexing.workers(2, 3 * indexing.BYTES_PER_WORKER) == 3


def test_batches_hold_at_most_so_many_files_and_bytes(monkeypatch):
    monkeypatch.setattr(indexing, "WRITE_BATCH", 3)
    monkeypatch.setattr(indexing, "BATCH_BYTES", 100)
    todo = [(f"f{i}", f"f{i}", 0, size) for i, size in enumerate([10, 10, 10, 10, 95, 200, 1])]
    assert [[t[0] for t in b] for b in indexing._batches(todo)] == [["f0", "f1", "f2"], ["f3"], ["f4"], ["f5"], ["f6"]]
    assert indexing._batches([]) == []
