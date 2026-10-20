import os

import pytest

from sift import store

from .conftest import write


def doc_ids(db):
    return dict(db.execute("SELECT path, id FROM docs"))


def pieces(text, first_line=1):
    """Chunks as (first line, last line, text): chunk() gives spans of the text."""
    return [(first, last, text[pos : pos + length]) for first, last, pos, length in store.chunk(text, first_line)]


def loaded(path):
    """The one document load() reads from a Markdown or text file, with its chunks as text."""
    (doc,), problems = store.load(path)
    assert problems == []
    return doc._replace(chunks=[(a, b, doc.text[p : p + n]) for a, b, p, n in doc.chunks])


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
    monkeypatch.setattr(store.time, "time_ns", lambda: 1)  # before Python 3.13 the Windows clock ticks every 15.6 ms
    store.touch(db)
    before = store.get_meta(db, "version")
    store.touch(db)
    assert store.get_meta(db, "version") != before


# --- parsing -----------------------------------------------------------------


def test_parse_front_matter():
    text = '---\ntitle: "A title"\ntags: ["a", "b"]\nyear: 2024\nnote: plain\nq: \'x\'\n  nested: no\n---\n# H\n'
    title, meta, body, first_line = store.parse(text, "fallback")
    assert title == "A title"
    assert meta == {"title": "A title", "tags": ["a", "b"], "year": 2024, "note": "plain", "q": "x"}
    assert body == "# H\n"
    assert first_line == 9


def test_parse_keeps_json_unsafe_constants_as_text():
    _, meta, _, _ = store.parse("---\na: NaN\nb: [1, Infinity]\n---\n", "f")
    assert meta == {"a": "NaN", "b": [1, "Infinity"]}


def test_parse_windows_line_endings():
    title, meta, body, first_line = store.parse("---\r\ntitle: T\r\n---\r\nbody\r\n", "f")
    assert (title, meta, body, first_line) == ("T", {"title": "T"}, "body\r\n", 4)


def test_parse_title_fallbacks():
    assert store.parse("intro\n## Second level ##\n", "file")[0] == "Second level"
    assert store.parse("no heading here\n", "file")[0] == "file"
    assert store.parse('---\ntitle: ""\n---\n# From heading\n', "file")[0] == "From heading"
    assert store.parse("---\ntitle: 42\n---\nbody\n", "file")[0] == "file"


def test_parse_without_closing_front_matter_is_plain_text():
    title, meta, body, _ = store.parse("---\ntitle: x\nno end\n", "file")
    assert (title, meta) == ("file", {})
    assert body.startswith("---")


# --- chunking ----------------------------------------------------------------


def test_chunk_short_text_is_one_chunk():
    assert list(pieces("# T\n\nshort body\n", 5)) == [(5, 7, "# T\n\nshort body")]


@pytest.mark.parametrize("text", ["", "\n\n  \n"])
def test_chunk_empty_text(text):
    assert list(pieces(text)) == []


def test_chunk_splits_with_line_numbers_and_overlap():
    text = "".join(f"line {i} " + "x" * 90 + "\n" for i in range(100))
    chunks = list(pieces(text))
    assert len(chunks) > 1
    assert chunks[0][0] == 1
    assert chunks[-1][1] == 100
    assert all(len(c[2]) <= store.CHUNK_CHARS for c in chunks)
    assert chunks[1][0] <= chunks[0][1]  # consecutive chunks overlap


def test_chunk_prefers_headings_without_overlap():
    text = "# One\n" + "a " * 900 + "\n# Two\nsecond section\n"
    chunks = list(pieces(text))
    assert [c[2].splitlines()[0] for c in chunks] == ["# One", "# Two"]
    assert chunks[1][0] == chunks[0][1] + 1


def test_chunk_never_exceeds_limit_with_long_lines():
    text = "short line\n" * 20 + "y" * (store.CHUNK_CHARS - 50) + "\n" + "z " * 3000
    assert all(len(c[2]) <= store.CHUNK_CHARS for c in pieces(text))


def test_chunk_breaks_unbroken_text():
    chunks = list(pieces("x" * (store.CHUNK_CHARS * 2 + 10)))
    assert [len(c[2]) for c in chunks] == [store.CHUNK_CHARS, store.CHUNK_CHARS, 10]


# --- files -------------------------------------------------------------------


def test_load(tmp_path):
    path = tmp_path / "note.md"
    path.write_bytes("﻿---\ntitle: BOM\n---\nbody \xe9\n".encode())
    doc = loaded(path)
    assert (doc.key, doc.title, doc.meta, doc.chunks) == ("", "BOM", {"title": "BOM"}, [(4, 4, "body \xe9")])
    assert doc.text == "---\ntitle: BOM\n---\nbody \xe9\n"  # the whole file, as get returns it
    assert len(doc.digest) == 32
    assert store.load(tmp_path / "missing.md") is None


def test_load_normalises_windows_line_endings(tmp_path):
    path = tmp_path / "crlf.md"
    path.write_bytes(b"---\r\ntitle: T\r\n---\r\n# T\r\n\r\nbody\r\n")
    assert loaded(path).chunks == [(4, 6, "# T\n\nbody")]


def test_walk_skips_hidden_and_vendor_dirs(notes):
    found = sorted(os.path.relpath(p, notes).replace(os.sep, "/") for p, _, _ in store.walk(str(notes)))
    assert found == ["plain.txt", "rl.md", "robots/brush.md"]


def test_walk_survives_missing_root_and_unstatable_files(tmp_path, monkeypatch):
    assert list(store.walk(str(tmp_path / "missing"))) == []

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

    monkeypatch.setattr(store.os, "scandir", lambda path: Entries([Entry()]))
    assert list(store.walk("anywhere")) == []


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
    store.index(db)
    after = doc_ids(db)
    assert set(after) == {"rl.md", "robots/brush.md", "new.md"}
    assert after["robots/brush.md"] == ids["robots/brush.md"]
    assert after["rl.md"] != ids["rl.md"]
    assert db.execute("SELECT mtime FROM sources WHERE path = 'robots/brush.md'").fetchone() == (10**18,)
    assert "1 added, 1 changed, 1 removed" in capsys.readouterr().err
    orphans = db.execute("SELECT count(*) FROM chunks WHERE doc NOT IN (SELECT id FROM docs)").fetchone()[0]
    assert orphans == 0


def test_index_in_several_batches(notes, home, monkeypatch):
    monkeypatch.setattr(store, "WRITE_BATCH", 2)
    db = store.connect(home)
    store.add_collection(db, notes, "notes")
    store.index(db)
    assert len(doc_ids(db)) == 3


def index_contents(db):
    """Everything indexing produces, independent of row ids."""
    return sorted(
        db.execute("""SELECT d.path, d.hash, d.title, d.meta, c.seq, c.line_start, c.line_end, t.text
                      FROM docs d JOIN chunks c ON c.doc = d.id JOIN chunk_text t ON t.id = c.id""")
    )


@pytest.fixture
def pools(monkeypatch):
    """Records every worker process pool that indexing starts."""
    created = []

    class RecordingPool(store.ProcessPoolExecutor):
        def __init__(self, max_workers, mp_context=None):
            created.append((max_workers, mp_context and mp_context.get_start_method()))
            super().__init__(max_workers, mp_context=mp_context)

    monkeypatch.setattr(store, "ProcessPoolExecutor", RecordingPool)
    return created


def test_index_parses_in_worker_processes(notes, tmp_path, monkeypatch, pools):
    serial = store.connect(tmp_path / "serial")
    store.add_collection(serial, notes, "notes")
    store.index(serial)
    assert pools == []

    monkeypatch.setattr(store, "_cpus", lambda: 3)  # two workers, one CPU for writing
    monkeypatch.setattr(store, "FILES_PER_WORKER", 1)
    monkeypatch.setattr(store, "WRITE_BATCH", 2)  # several batches in flight
    parallel = store.connect(tmp_path / "parallel")
    store.add_collection(parallel, notes, "notes")
    store.index(parallel)
    # spawned, not forked, on every platform: forking a process with threads (PyTorch starts some) can deadlock
    assert pools == [(2, "spawn")]
    assert index_contents(parallel) == index_contents(serial)


def test_small_updates_skip_worker_processes(corpus, monkeypatch, pools):
    db, root = corpus
    monkeypatch.setattr(store, "_cpus", lambda: 64)
    write(root / "new.md", "# New\n")
    store.index(db)
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
    monkeypatch.setattr(store, "_cpus", lambda: cpus)
    assert store.workers(files) == expected


def test_cpus_respects_what_the_platform_allows(monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 6, raising=False)  # Python 3.13+
    assert store._cpus() == 6
    monkeypatch.delattr(os, "process_cpu_count")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)  # Linux
    assert store._cpus() == 3
    monkeypatch.delattr(os, "sched_getaffinity")
    monkeypatch.setattr(os, "cpu_count", lambda: None)  # unknown
    assert store._cpus() == 1


def test_index_selected_collections(corpus, tmp_path):
    db, _ = corpus
    other = tmp_path / "other"
    write(other / "a.md", "# A\n")
    store.add_collection(db, other, "other")
    store.index(db, ["notes"])
    assert "a.md" not in doc_ids(db)
    with pytest.raises(SystemExit, match="'nope'"):
        store.index(db, ["nope"])


def test_index_skips_missing_folder_without_deleting(corpus, capsys):
    db, root = corpus
    root.rename(root.with_name("moved"))
    store.index(db)
    assert len(doc_ids(db)) == 3
    assert "folder not found" in capsys.readouterr().err


def test_index_skips_unreadable_files_and_retries_later(notes, home, monkeypatch, capsys):
    real_load = store.load
    monkeypatch.setattr(store, "load", lambda path, fields=None: None if path.endswith("rl.md") else real_load(path))
    db = store.connect(home)
    store.add_collection(db, notes, "notes")
    store.index(db)
    assert "rl.md" not in doc_ids(db)
    assert "1 unreadable" in capsys.readouterr().err
    monkeypatch.setattr(store, "load", real_load)
    store.index(db)
    assert "rl.md" in doc_ids(db)


# --- edge cases: titles, front matter, lines, encodings, names -----------------


@pytest.mark.parametrize(
    "text, title",
    [
        ("# Learning C#\n", "Learning C#"),  # a closing sequence of #s needs a space before it
        ("## Second level ##\n", "Second level"),
        ("#\nA paragraph, not a heading\n# Real\n", "Real"),  # an empty heading names nothing
        ("# ##\n# Real\n", "Real"),
        ("   # Indented up to three spaces\n", "Indented up to three spaces"),
        ("    # Four spaces make it code\n", "file"),
        ("#hashtag\n", "file"),
        ("Intro\n\n```bash\n# install it\n```\n# After the code\n", "After the code"),  # no headings in code
        ("~~~\n# not a heading\n~~~\n", "file"),
        ("````\n```\n# still code\n````\n", "file"),  # a fence closes only with its own kind and length
        ("```\n# unclosed code block\n", "file"),
    ],
)
def test_title_from_the_first_markdown_heading(text, title):
    assert store.parse(text, "file")[0] == title


def test_heading_search_stays_fast_on_long_lines():
    # A heading followed by a long run of spaces made the heading regex backtrack for minutes, stalling `sift index`.
    text = "# a" + " " * 100_000 + "b\n" + "#" + " " * 100_000 + "\n"
    assert store.parse(text, "file")[0] == "a" + " " * 100_000 + "b"


def test_parse_empty_front_matter():
    assert store.parse("---\n---\n# Title\nbody\n", "file") == ("Title", {}, "# Title\nbody\n", 3)


def test_parse_front_matter_lists_and_comments():
    # Obsidian, Jekyll and Hugo write lists as one `- item` per line.
    text = '---\n# a comment\ntags:\n  - painting\n  - "robots"\n- 2024\nauthors: ["A"]\nempty:\n---\nbody\n'
    assert store.parse(text, "file")[1] == {"tags": ["painting", "robots", 2024], "authors": ["A"], "empty": None}


@pytest.mark.parametrize("separator", ["\x0b", "\x0c", "\x1c", "\x85", "\u2028", "\u2029"])
def test_chunk_counts_lines_like_editors(separator):
    # Only \n ends a line; str.splitlines also splits at these, which shifted the line numbers search hits report.
    assert list(pieces(f"one\n{separator}two\nthree\n")) == [(1, 3, f"one\n{separator}two\nthree")]


@pytest.mark.parametrize("bom, codec", [(b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be")])
def test_load_reads_utf16_files(tmp_path, bom, codec):
    # Windows PowerShell 5.1 writes UTF-16 with `>` and Out-File; read as UTF-8, every other character was a NUL.
    path = tmp_path / "note.txt"
    path.write_bytes(bom + "# Héllo\r\nworld\r\n".encode(codec))
    doc = loaded(path)
    assert (doc.title, doc.meta, doc.chunks) == ("Héllo", {}, [(1, 2, "# Héllo\nworld")])


def test_load_replaces_invalid_utf8(tmp_path):
    path = tmp_path / "latin1.txt"
    path.write_bytes(b"r\xe9sum\xe9 du jour\n")
    assert loaded(path).chunks == [(1, 1, "r\ufffdsum\ufffd du jour")]


def test_index_skips_file_names_sqlite_cannot_store(corpus, monkeypatch, capsys):
    # Linux file names may hold bytes that aren't UTF-8, which Python keeps as surrogates and SQLite refuses:
    # one such file aborted the whole run.
    db, root = corpus
    real_walk = store.walk

    def walk(path):
        yield os.path.join(path, "r\udce9sum\udce9.md"), 1, 1
        yield from real_walk(path)

    monkeypatch.setattr(store, "walk", walk)
    write(root / "new.md", "# New\n")
    store.index(db)
    assert "new.md" in doc_ids(db)
    assert "unsupported file name, skipped: " in capsys.readouterr().err


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


def test_a_few_big_files_still_get_workers(monkeypatch):
    # 424 JSON Lines files of arXiv are few files but gigabytes of records to parse
    monkeypatch.setattr(store, "_cpus", lambda: 16)
    assert store.workers(424, 0) == 0
    assert store.workers(424, 5 << 30) == 15
    assert store.workers(2, 3 * store.BYTES_PER_WORKER) == 3


def test_batches_hold_at_most_so_many_files_and_bytes(monkeypatch):
    monkeypatch.setattr(store, "WRITE_BATCH", 3)
    monkeypatch.setattr(store, "BATCH_BYTES", 100)
    todo = [(f"f{i}", f"f{i}", 0, size) for i, size in enumerate([10, 10, 10, 10, 95, 200, 1])]
    assert [[t[0] for t in b] for b in store._batches(todo)] == [["f0", "f1", "f2"], ["f3"], ["f4"], ["f5"], ["f6"]]
    assert store._batches([]) == []
