import os

import pytest

from sift import store

from .conftest import write


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
    db, _ = corpus
    store.remove_collection(db, "notes")
    for table in ("collections", "docs", "chunks", "fts", "vectors"):
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
    assert list(store.chunk("# T\n\nshort body\n", 5)) == [(5, 7, "# T\n\nshort body")]


@pytest.mark.parametrize("text", ["", "\n\n  \n"])
def test_chunk_empty_text(text):
    assert list(store.chunk(text)) == []


def test_chunk_splits_with_line_numbers_and_overlap():
    text = "".join(f"line {i} " + "x" * 90 + "\n" for i in range(100))
    chunks = list(store.chunk(text))
    assert len(chunks) > 1
    assert chunks[0][0] == 1 and chunks[-1][1] == 100
    assert all(len(c[2]) <= store.CHUNK_CHARS for c in chunks)
    assert chunks[1][0] <= chunks[0][1]  # consecutive chunks overlap


def test_chunk_prefers_headings_without_overlap():
    text = "# One\n" + "a " * 900 + "\n# Two\nsecond section\n"
    chunks = list(store.chunk(text))
    assert [c[2].splitlines()[0] for c in chunks] == ["# One", "# Two"]
    assert chunks[1][0] == chunks[0][1] + 1


def test_chunk_never_exceeds_limit_with_long_lines():
    text = "short line\n" * 20 + "y" * (store.CHUNK_CHARS - 50) + "\n" + "z " * 3000
    assert all(len(c[2]) <= store.CHUNK_CHARS for c in store.chunk(text))


def test_chunk_breaks_unbroken_text():
    chunks = list(store.chunk("x" * (store.CHUNK_CHARS * 2 + 10)))
    assert [len(c[2]) for c in chunks] == [store.CHUNK_CHARS, store.CHUNK_CHARS, 10]


# --- files -------------------------------------------------------------------


def test_load(tmp_path):
    path = tmp_path / "note.md"
    path.write_bytes("﻿---\ntitle: BOM\n---\nbody \xe9\n".encode())
    digest, title, meta, chunks = store.load(path)
    assert (title, meta, chunks) == ("BOM", {"title": "BOM"}, [(4, 4, "body \xe9")])
    assert len(digest) == 32
    assert store.load(tmp_path / "missing.md") is None


def test_load_normalises_windows_line_endings(tmp_path):
    path = tmp_path / "crlf.md"
    path.write_bytes(b"---\r\ntitle: T\r\n---\r\n# T\r\n\r\nbody\r\n")
    assert store.load(path)[3] == [(4, 6, "# T\n\nbody")]


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
    assert db.execute("SELECT mtime FROM docs WHERE path = 'robots/brush.md'").fetchone() == (10**18,)
    assert "1 added, 1 changed, 1 removed" in capsys.readouterr().err
    orphans = db.execute("SELECT count(*) FROM chunks WHERE doc NOT IN (SELECT id FROM docs)").fetchone()[0]
    assert orphans == 0


def test_index_in_several_batches(notes, home, monkeypatch):
    monkeypatch.setattr(store, "WRITE_BATCH", 2)
    db = store.connect(home)
    store.add_collection(db, notes, "notes")
    store.index(db)
    assert len(doc_ids(db)) == 3


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
    monkeypatch.setattr(store, "load", lambda path: None if path.endswith("rl.md") else real_load(path))
    db = store.connect(home)
    store.add_collection(db, notes, "notes")
    store.index(db)
    assert "rl.md" not in doc_ids(db)
    assert "1 unreadable" in capsys.readouterr().err
    monkeypatch.setattr(store, "load", real_load)
    store.index(db)
    assert "rl.md" in doc_ids(db)
