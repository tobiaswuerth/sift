"""JSON Lines files: one document per line, with id, title and text found by name and the rest as metadata."""

import json
import sqlite3

import pytest

from sift import store
from sift.search import Index

from .conftest import write

PAPERS = [
    {
        "id": "2101.00001",
        "title": "Brush stroke planning",
        "abstract": "A robot plans brush strokes.",
        "year": 2021,
        "authors": ["Ann Lee", "Bo Chen"],
    },
    {
        "id": "2101.00002",
        "title": "Reward shaping",
        "abstract": "Shaping speeds up reinforcement learning.",
        "year": 2021,
    },
    {"id": "hep-th/9901001", "title": "String junctions", "abstract": "Heterotic strings and junctions.", "year": 1999},
]


def jsonl(records):
    return "".join(json.dumps(r) + "\n" for r in records)


@pytest.fixture
def papers(tmp_path, home):
    root = tmp_path / "arxiv"
    write(root / "2021" / "2021-01.jsonl", jsonl(PAPERS[:2]))
    write(root / "1999" / "1999-01.jsonl", jsonl(PAPERS[2:]))
    write(root / "notes.md", "# Notes\n\nA Markdown file in the same collection.\n")
    db = store.connect(home)
    store.add_collection(db, root, "arxiv")
    store.index(db)
    return db, root


def docs(db):
    return dict(db.execute("SELECT path, title FROM docs"))


def test_each_line_is_a_document(papers):
    db, _ = papers
    assert docs(db) == {
        "2021/2021-01/2101.00001": "Brush stroke planning",
        "2021/2021-01/2101.00002": "Reward shaping",
        "1999/1999-01/hep-th/9901001": "String junctions",
        "notes.md": "Notes",
    }


def test_the_text_is_title_and_abstract_and_everything_else_metadata(papers):
    db, _ = papers
    text, meta = db.execute("SELECT text, meta FROM docs WHERE path = '2021/2021-01/2101.00001'").fetchone()
    assert text == "# Brush stroke planning\n\nA robot plans brush strokes."
    assert json.loads(meta) == {
        "id": "2101.00001",
        "title": "Brush stroke planning",
        "year": 2021,
        "authors": ["Ann Lee", "Bo Chen"],
    }


def test_search_filters_fields_and_get(papers):
    db, _ = papers
    index = Index(db)
    hit = index.search("brush", fields=["year", "authors"])[0]
    assert (hit["id"], hit["lines"], hit["fields"]) == (
        "arxiv/2021/2021-01/2101.00001",
        "1-3",
        {"year": 2021, "authors": ["Ann Lee", "Bo Chen"]},
    )
    assert [h["id"] for h in index.search("strings OR brush", filters={"year": {"<": 2000}})] == [
        "arxiv/1999/1999-01/hep-th/9901001"
    ]
    assert [h["id"] for h in index.search("brush", filters={"authors": "Bo Chen"})] == ["arxiv/2021/2021-01/2101.00001"]
    assert index.get("arxiv/2021/2021-01/2101.00002") == "# Reward shaping\n\nShaping speeds up reinforcement learning."
    assert index.get("arxiv/hep-th/9901001").startswith("# String junctions")  # by the record id alone
    assert sorted(h["id"] for h in index.search("brush OR shaping OR strings", path="2021/*")) == [
        "arxiv/2021/2021-01/2101.00001",
        "arxiv/2021/2021-01/2101.00002",
    ]  # folders still work as globs


@pytest.mark.parametrize(
    "record, title, text",
    [
        ({"id": 1, "title": "T", "text": "body", "abstract": "not this"}, "T", "# T\n\nbody"),
        ({"_id": "a", "name": "N", "content": "body"}, "N", "# N\n\nbody"),
        ({"id": 1, "headline": "H", "body": ["one", "two"]}, "H", "# H\n\none\n\ntwo"),  # a list of strings
        ({"id": 1, "subject": "S", "description": "body"}, "S", "# S\n\nbody"),
        ({"id": 1, "summary": "# Own heading\n\nbody"}, "Own heading", "# Own heading\n\nbody"),  # no title field
        (
            {"id": 1, "question": "short", "answer": "the longest string is the text"},
            "1",
            "the longest string is the text",
        ),  # no known text field
    ],
)
def test_fields_are_found_by_name(tmp_path, home, record, title, text):
    write(tmp_path / "c" / "r.jsonl", json.dumps(record) + "\n")
    db = store.connect(home)
    store.add_collection(db, tmp_path / "c", "c")
    store.index(db)
    assert db.execute("SELECT title, text FROM docs").fetchone() == (title, text)


def test_records_without_an_id_are_numbered_by_line(tmp_path, home):
    write(tmp_path / "c" / "r.jsonl", jsonl([{"text": "first"}, {"text": "second"}]))
    db = store.connect(home)
    store.add_collection(db, tmp_path / "c", "c")
    store.index(db)
    assert sorted(docs(db)) == ["r/1", "r/2"]


def test_fields_chosen_when_adding_the_collection(tmp_path, home):
    write(tmp_path / "c" / "qa.jsonl", json.dumps({"key": "q1", "q": "Why?", "a": "Because.", "text": "not this"}))
    db = store.connect(home)
    store.add_collection(db, tmp_path / "c", "c", fields={"id": "key", "title": "q", "text": ["q", "a"]})
    store.index(db)
    assert db.execute("SELECT path, title, text FROM docs").fetchone() == (
        "qa/q1",
        "Why?",
        "# Why?\n\nWhy?\n\nBecause.",
    )


@pytest.mark.parametrize("fields", [{"text": []}, {"text": [""]}, {"id": ""}, {"title": 3}, {"other": "x"}])
def test_bad_field_choices_are_clear_errors(tmp_path, home, fields):
    (tmp_path / "c").mkdir()
    with pytest.raises(SystemExit, match="field"):
        store.add_collection(store.connect(home), tmp_path / "c", "c", fields=fields)


def test_bad_lines_are_skipped_and_reported(tmp_path, home, capsys):
    lines = [
        "{not json",
        "[1, 2]",
        json.dumps({"id": 1}),
        json.dumps({"id": 2, "text": "kept"}),
        "",
        json.dumps({"id": 2, "text": "duplicate"}),
    ]
    write(tmp_path / "c" / "r.jsonl", "\n".join(lines) + "\n")
    db = store.connect(home)
    store.add_collection(db, tmp_path / "c", "c")
    store.index(db)
    assert db.execute("SELECT path, text FROM docs").fetchall() == [("r/2", "kept")]
    err = capsys.readouterr().err
    assert "4 lines skipped" in err
    for reason in ("line 1: not JSON", "line 2: not an object", "line 3: no text", "line 6: id 2 used twice"):
        assert reason in err


# --- incremental indexing: unchanged files and records cost nothing ----------------


def doc_ids(db):
    return dict(db.execute("SELECT path, id FROM docs"))


def test_an_unchanged_file_is_not_read_again(papers, monkeypatch):
    db, _ = papers
    read = []
    monkeypatch.setattr(store, "load", lambda path, fields=None: read.append(path))
    store.index(db)
    assert read == []


def test_only_changed_records_are_indexed_again(papers, capsys):
    db, root = papers
    before = doc_ids(db)
    changed = [
        PAPERS[0],
        dict(PAPERS[1], abstract="Shaping is now different."),
        {"id": "2101.00003", "title": "New", "abstract": "A new paper."},
    ]
    write(root / "2021" / "2021-01.jsonl", jsonl(changed))
    capsys.readouterr()
    store.index(db)
    after = doc_ids(db)
    assert after["2021/2021-01/2101.00001"] == before["2021/2021-01/2101.00001"]  # untouched
    assert after["2021/2021-01/2101.00002"] != before["2021/2021-01/2101.00002"]
    assert "1 added, 1 changed, 0 removed" in capsys.readouterr().err
    assert [h["id"] for h in Index(db).search("shaping")] == ["arxiv/2021/2021-01/2101.00002"]
    assert Index(db).search("speeds") == []  # the old text is gone from the keyword index


def test_removed_records_and_files_leave_the_index(papers, capsys):
    db, root = papers
    write(root / "2021" / "2021-01.jsonl", jsonl(PAPERS[:1]))
    (root / "1999" / "1999-01.jsonl").unlink()
    capsys.readouterr()
    store.index(db)
    assert sorted(docs(db)) == ["2021/2021-01/2101.00001", "notes.md"]
    assert "0 added, 0 changed, 2 removed" in capsys.readouterr().err
    assert db.execute("SELECT count(*) FROM sources").fetchone() == (2,)
    assert Index(db).search("junctions") == []


def test_vectors_of_unchanged_records_are_kept(papers, fake_models):
    db, root = papers
    fake_models.embed_pending(db)
    write(root / "2021" / "2021-01.jsonl", jsonl([PAPERS[0], dict(PAPERS[1], abstract="New text.")]))
    store.index(db)
    pending = db.execute("SELECT count(*) FROM chunks WHERE id NOT IN (SELECT chunk FROM vectors)").fetchone()
    assert pending == (1,)  # only the changed record needs embedding


# --- the index holds the text: chunks are spans of it, stored once ------------------


def test_chunks_are_spans_of_the_stored_text(tmp_path, home):
    long = "\n".join(f"Line {i} " + "word " * 40 for i in range(200))
    write(tmp_path / "c" / "long.md", "---\ntitle: Long\n---\n" + long + "\n")
    write(tmp_path / "c" / "r.jsonl", json.dumps({"id": 1, "title": "R", "text": long}) + "\n")
    db = store.connect(home)
    store.add_collection(db, tmp_path / "c", "c")
    store.index(db)
    rows = db.execute("""SELECT d.text, c.pos, c.len, c.line_start, c.line_end, t.text FROM chunks c
                         JOIN docs d ON d.id = c.doc JOIN chunk_text t ON t.id = c.id""").fetchall()
    assert len(rows) > 4
    for text, pos, length, first, last, chunk in rows:
        assert chunk == text[pos : pos + length]
        assert chunk.splitlines()[0] in text.split("\n")[first - 1]  # the line numbers match the span
        assert chunk.splitlines()[-1] in text.split("\n")[last - 1]
        assert len(chunk) <= store.CHUNK_CHARS


def test_get_returns_the_indexed_text_until_the_next_index(corpus):
    # Hits and get always agree: both come from the index, not from the file as it is now.
    db, root = corpus
    write(root / "rl.md", "# Rewritten\n")
    assert Index(db).get("notes/rl.md").startswith("# Reward shaping")
    (root / "rl.md").unlink()
    store.index(db)
    with pytest.raises(LookupError, match="unknown document"):
        Index(db).get("notes/rl.md")


# --- the database layout has a version -------------------------------------------


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


def test_a_record_named_like_a_file_is_skipped(tmp_path, home, capsys):
    # a.jsonl's record "b.md" and the file a/b.md would both be a/b.md: one of them must give way
    write(tmp_path / "c" / "a" / "b.md", "# File\n")
    write(tmp_path / "c" / "a.jsonl", json.dumps({"id": "b.md", "text": "record"}) + "\n")
    db = store.connect(home)
    store.add_collection(db, tmp_path / "c", "c")
    store.index(db)
    assert db.execute("SELECT count(*) FROM docs WHERE path = 'a/b.md'").fetchone() == (1,)
    assert "a/b.md: a document of this name exists already, skipped" in capsys.readouterr().err
