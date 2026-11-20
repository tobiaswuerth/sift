"""JSON Lines files: one document per line, with id, title and text found by name and the rest as metadata."""

import io
import json
import random
import re
import sqlite3
from pathlib import Path

import pytest

from sift import embed, indexing, readers, store
from sift.catalog import Catalog
from sift.search import Index

from .conftest import index_contents, keywords_agree, only, prose_text, write

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
    catalog = Catalog(home)
    catalog.add(root, "arxiv")
    indexing.index(catalog)
    return catalog, root


def docs(catalog):
    return dict(only(catalog).execute("SELECT path, title FROM docs"))


def test_each_line_is_a_document(papers):
    catalog, _ = papers
    assert docs(catalog) == {
        "2021/2021-01/2101.00001": "Brush stroke planning",
        "2021/2021-01/2101.00002": "Reward shaping",
        "1999/1999-01/hep-th/9901001": "String junctions",
        "notes.md": "Notes",
    }


def test_the_text_is_title_and_abstract_and_everything_else_metadata(papers):
    catalog, _ = papers
    text, meta = only(catalog).execute("SELECT text, meta FROM docs WHERE path = '2021/2021-01/2101.00001'").fetchone()
    assert text == "# Brush stroke planning\n\nA robot plans brush strokes."
    assert json.loads(meta) == {
        "id": "2101.00001",
        "title": "Brush stroke planning",
        "year": 2021,
        "authors": ["Ann Lee", "Bo Chen"],
    }


def test_search_filters_fields_and_get(papers):
    catalog, _ = papers
    index = Index(catalog)
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
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    assert only(catalog).execute("SELECT title, text FROM docs").fetchone() == (title, text)


def test_records_without_an_id_are_numbered_by_line(tmp_path, home):
    write(tmp_path / "c" / "r.jsonl", jsonl([{"text": "first"}, {"text": "second"}]))
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    assert sorted(docs(catalog)) == ["r/1", "r/2"]


def test_fields_chosen_when_adding_the_collection(tmp_path, home):
    write(tmp_path / "c" / "qa.jsonl", json.dumps({"key": "q1", "q": "Why?", "a": "Because.", "text": "not this"}))
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c", fields={"id": "key", "title": "q", "text": ["q", "a"]})
    indexing.index(catalog)
    assert only(catalog).execute("SELECT path, title, text FROM docs").fetchone() == (
        "qa/q1",
        "Why?",
        "# Why?\n\nWhy?\n\nBecause.",
    )


@pytest.mark.parametrize("fields", [{"text": []}, {"text": [""]}, {"id": ""}, {"title": 3}, {"other": "x"}])
def test_bad_field_choices_are_clear_errors(tmp_path, home, fields):
    (tmp_path / "c").mkdir()
    with pytest.raises(SystemExit, match="field"):
        Catalog(home).add(tmp_path / "c", "c", fields=fields)


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
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    assert only(catalog).execute("SELECT path, text FROM docs").fetchall() == [("r/2", "kept")]
    err = capsys.readouterr().err
    assert "4 lines skipped" in err
    for reason in ("line 1: not JSON", "line 2: not an object", "line 3: no text", "line 6: id 2 used twice"):
        assert reason in err


def test_lines_sqlite_cant_store_or_python_cant_nest_stop_nothing(tmp_path, home, capsys, json_room):
    # Exported chats hold an emoji's half alone ("\ud83d"), which SQLite refuses; a line nested a hundred thousand deep
    # Python can't read. Either stopped `sift index`, for every collection after it too, on every run.
    # SQLite stores JSON nested 1,000 deep at most: a line nested 1,100 deep stopped it all the same.
    lines = [
        '{"id": "1", "text": "a \\ud83d b"}',
        "[" * 100_000 + "]" * 100_000,
        json.dumps({"id": "3", "text": "kept"}),
        '{"id": "4", "text": "x", "v": ' + "[" * 1100 + "1" + "]" * 1100 + "}",
        '{"id": "5", "text": "near the limit", "v": ' + "[" * 998 + "1" + "]" * 998 + "}",
    ]
    write(tmp_path / "c" / "chat.jsonl", "\n".join(lines) + "\n")
    write(tmp_path / "c" / "note.md", "# Note\n\nafter the chat\n")
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    assert dict(only(catalog).execute("SELECT path, text FROM docs")) == {
        "chat/1": "a \ufffd b",
        "chat/3": "kept",
        "chat/5": "near the limit",
        "note.md": "# Note\n\nafter the chat\n",
    }
    err = capsys.readouterr().err
    assert "line 2: nested too deeply" in err
    assert "line 4: nested too deeply" in err


def test_numbers_too_large_for_a_float_are_kept_as_written(tmp_path, home):
    # 1e999 read as infinity, which JSON can't hold: written back as Infinity, older SQLite refused the line's
    # front matter, and stopped `sift index`.
    write(tmp_path / "c" / "big.jsonl", '{"id": "1", "text": "big", "size": 1e999, "small": -1e-999, "n": 2.5}\n')
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    (meta,) = only(catalog).execute("SELECT meta FROM docs").fetchone()
    assert json.loads(meta) == {"id": "1", "size": "1e999", "small": -0.0, "n": 2.5}
    assert "Infinity" not in meta


# --- incremental indexing: unchanged files and records cost nothing ----------------


def doc_ids(catalog):
    return dict(only(catalog).execute("SELECT path, id FROM docs"))


def test_an_unchanged_file_is_not_read_again(papers, monkeypatch):
    catalog, _ = papers
    read = []
    monkeypatch.setattr(readers, "load", lambda path, fields=None: read.append(path))
    indexing.index(catalog)
    assert read == []


def test_only_changed_records_are_indexed_again(papers, capsys):
    catalog, root = papers
    before = doc_ids(catalog)
    changed = [
        PAPERS[0],
        dict(PAPERS[1], abstract="Shaping is now different."),
        {"id": "2101.00003", "title": "New", "abstract": "A new paper."},
    ]
    write(root / "2021" / "2021-01.jsonl", jsonl(changed))
    capsys.readouterr()
    indexing.index(catalog)
    after = doc_ids(catalog)
    assert after["2021/2021-01/2101.00001"] == before["2021/2021-01/2101.00001"]  # untouched, the same row
    # The changed one by its text, not its row id: ids never repeat (AUTOINCREMENT, see test_store), but which record
    # gets which new id depends on the order the OS lists files in.
    text = only(catalog).execute("SELECT text FROM docs WHERE path = '2021/2021-01/2101.00002'").fetchone()[0]
    assert text == "# Reward shaping\n\nShaping is now different."
    assert "1 added, 1 changed, 0 removed" in capsys.readouterr().err
    assert [h["id"] for h in Index(catalog).search("shaping")] == ["arxiv/2021/2021-01/2101.00002"]
    assert Index(catalog).search("speeds") == []  # the old text is gone from the keyword index


def test_removed_records_and_files_leave_the_index(papers, capsys):
    catalog, root = papers
    write(root / "2021" / "2021-01.jsonl", jsonl(PAPERS[:1]))
    (root / "1999" / "1999-01.jsonl").unlink()
    capsys.readouterr()
    indexing.index(catalog)
    assert sorted(docs(catalog)) == ["2021/2021-01/2101.00001", "notes.md"]
    assert "0 added, 0 changed, 2 removed" in capsys.readouterr().err
    assert only(catalog).execute("SELECT count(*) FROM sources").fetchone() == (2,)
    assert Index(catalog).search("junctions") == []


def test_vectors_of_unchanged_records_are_kept(papers, fake_models):
    catalog, root = papers
    embed.embed_pending(catalog)
    write(root / "2021" / "2021-01.jsonl", jsonl([PAPERS[0], dict(PAPERS[1], abstract="New text.")]))
    indexing.index(catalog)
    pending = (
        only(catalog).execute("SELECT count(*) FROM chunks WHERE id NOT IN (SELECT chunk FROM vectors)").fetchone()
    )
    assert pending == (1,)  # only the changed record needs embedding


# --- the index holds the text: chunks are spans of it, stored once ------------------


def test_chunks_are_spans_of_the_stored_text(tmp_path, home):
    long = "\n".join(f"Line {i} " + "word " * 40 for i in range(200))
    write(tmp_path / "c" / "long.md", "---\ntitle: Long\n---\n" + long + "\n")
    write(tmp_path / "c" / "r.jsonl", json.dumps({"id": 1, "title": "R", "text": long}) + "\n")
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    rows = (
        only(catalog)
        .execute("""SELECT d.text, c.pos, c.len, c.line_start, c.line_end, t.text FROM chunks c
                         JOIN docs d ON d.id = c.doc JOIN chunk_text t ON t.id = c.id""")
        .fetchall()
    )
    assert len(rows) > 4
    for text, pos, length, first, last, chunk in rows:
        assert chunk == text[pos : pos + length]
        assert chunk.splitlines()[0] in text.split("\n")[first - 1]  # the line numbers match the span
        assert chunk.splitlines()[-1] in text.split("\n")[last - 1]
        assert len(chunk) <= readers.CHUNK_CHARS


def test_a_record_and_a_file_of_one_path_the_first_read_keeps_it(tmp_path, home, capsys):
    # a.jsonl's record "b.md" and the file a/b.md would both be a/b.md: one of them must give way
    write(tmp_path / "c" / "a" / "b.md", "# File\n")
    write(tmp_path / "c" / "a.jsonl", json.dumps({"id": "b.md", "text": "record"}) + "\n")
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    assert only(catalog).execute("SELECT text FROM docs WHERE path = 'a/b.md'").fetchall() == [("record",)]  # first
    shown = [line.rsplit("\r", 1)[-1].rstrip() for line in capsys.readouterr().err.split("\n")]
    said = "c: a/b.md: a document of a.jsonl has this path; this one of a/b.md is left out until it is free"
    assert said in shown  # a line of its own


def test_a_record_whose_id_holds_a_nul_can_be_got(tmp_path, home):
    # Its path held the NUL, at which SQLite's substr stops: `get` by its id, or its folder, found nothing.
    write(tmp_path / "c" / "data.jsonl", json.dumps({"id": "a\x00b", "text": "record"}) + "\n")
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    assert Index(catalog).get("c/data/a\ufffdb") == "record"


def test_two_records_of_one_path_the_first_keeps_it(tmp_path, home, capsys):
    # A NUL in an id is stored as U+FFFD: ids that differ only so are two records of one path. The second replaced
    # the first when the two came in different batches of their file.
    other = "a" + chr(0xFFFD) + "b"
    write(tmp_path / "c" / "data.jsonl", jsonl([{"id": "a\x00b", "text": "first"}, {"id": other, "text": "second"}]))
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    assert texts(catalog) == {f"data/{other}": "first"}
    left = (
        f"c: data/{other}: a document of data.jsonl has this path; this one of data.jsonl is left out until it is free"
    )
    assert left in said(capsys)


@pytest.mark.parametrize("stored", [1, 7])
def test_two_records_of_one_path_in_any_batches(tmp_path, home, monkeypatch, stored):
    monkeypatch.setattr(indexing, "STORED", stored)
    other = "a" + chr(0xFFFD) + "b"
    write(tmp_path / "c" / "data.jsonl", jsonl([{"id": "a\x00b", "text": "first"}, {"id": other, "text": "second"}]))
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    assert texts(catalog) == {f"data/{other}": "first"}


# --- large files: read as they are stored -----------------------------------------


@pytest.fixture
def large(monkeypatch):
    """Every JSON Lines file counts as large: read as it is stored, seven documents at a time."""
    monkeypatch.setattr(readers, "STREAMED", 0)
    monkeypatch.setattr(indexing, "STORED", 7)


def numbered(n):
    return jsonl({"id": i, "text": f"record {i}"} for i in range(n))


def texts(catalog):
    return dict(only(catalog).execute("SELECT path, text FROM docs"))


def said(capsys):
    """What sift said, line by line, each as it stood last."""
    return [line.rsplit("\r", 1)[-1].rstrip() for line in capsys.readouterr().err.split("\n")]


def test_a_large_file_is_read_no_further_than_it_is_stored(tmp_path, home, large, monkeypatch):
    # Read whole, a file of 300,000 records, 69 MB, took 697 MB of memory: its bytes, its lines and every record's
    # document at once, ten times the file. It is read a batch of records ahead of what is stored, whatever its size.
    write(tmp_path / "c" / "big.jsonl", numbered(95))
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    read, stored, ahead, record = [], [], [], readers._record
    monkeypatch.setattr(readers, "_record", lambda *args: read.append(1) or record(*args))

    def storing(statement):
        if statement.startswith("INSERT INTO docs") and statement not in stored:  # said again as its trigger starts
            ahead.append(len(read) - len(stored))
            stored.append(statement)

    catalog.store("c").set_trace_callback(storing)
    try:
        indexing.index(catalog)
    finally:
        catalog.store("c").set_trace_callback(None)
    assert len(stored) == 95
    assert max(ahead) <= 7


def test_a_batch_of_long_records_ends_by_its_text(tmp_path, home, large, monkeypatch):
    # Seven records of megabytes each are megabytes too many at once: a batch ends by the text it holds as well.
    monkeypatch.setattr(indexing, "STORED_CHARS", 100)
    write(tmp_path / "c" / "big.jsonl", jsonl({"id": i, "text": "word " * 30} for i in range(6)))
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    batches, keywords = [], store.keywords
    monkeypatch.setattr(
        store, "keywords", lambda db, docs, **more: batches.append(len(docs)) or keywords(db, docs, **more)
    )
    indexing.index(catalog)
    assert batches == [1] * 6


def lines_of(rng, n):
    """Lines as files hold them: records with ids and without, some ids twice, and lines that are no record."""
    lines = []
    for i in range(n):
        if rng.random() < 0.15:
            lines.append(rng.choice(["{not json", "[1, 2]", "", json.dumps({"id": i}), "   "]))
            continue
        record = {"title": f"Title {rng.randint(0, 20)}", "text": prose_text(rng, 30), "year": 2000 + i % 5}
        if rng.random() < 0.8:
            record["id"] = f"r{rng.randint(0, n)}"
        lines.append(json.dumps(record, ensure_ascii=rng.random() < 0.5))
    return lines


@pytest.mark.parametrize(
    "encoding, end", [("utf-8", "\n"), ("utf-8-sig", "\r\n"), ("utf-16", "\n"), ("utf-16-be", "\r\n"), ("utf-32", "\n")]
)
def test_a_large_file_is_indexed_as_one_read_whole(tmp_path, monkeypatch, capsys, encoding, end):
    # The same documents, lines skipped and counts, to the character, when it is first indexed and when it changed.
    rng, root = random.Random(8), tmp_path / "c"
    root.mkdir()
    whole, streamed = Catalog(tmp_path / "whole"), Catalog(tmp_path / "streamed")
    for catalog in (whole, streamed):
        catalog.add(root, "c")
    monkeypatch.setattr(indexing, "STORED", 7)

    def indexed(lines):
        data = (end.join(lines) + end).encode(encoding)
        mark = b"\xfe\xff" if encoding == "utf-16-be" else b""  # the mark Python writes for no byte order asked
        torn = b'{"id": "torn", "text": "na\xefve \xff\xfe torn"}\n' if encoding == "utf-8" else b""  # not UTF-8
        (root / "big.jsonl").write_bytes(mark + data + torn)
        told = []
        for catalog, past in ((whole, 10**9), (streamed, 0)):
            monkeypatch.setattr(readers, "STREAMED", past)
            capsys.readouterr()
            indexing.index(catalog)
            told.append(
                sorted(
                    re.sub(r" \(\d+\.\ds\)$", "", line) for line in said(capsys) if "skipped" in line or "added" in line
                )
            )
            assert keywords_agree(catalog.store("c"))
        assert told[0] == told[1]
        assert index_contents(whole.store("c")) == index_contents(streamed.store("c"))
        return told[0]

    lines = lines_of(rng, 120)
    first = indexed(lines)
    assert any("lines skipped" in line for line in first)
    changed = [line.replace("Title", "Changed") if rng.random() < 0.2 else line for line in lines if rng.random() > 0.2]
    again = indexed(changed + lines_of(rng, 30))
    assert any(re.search(r"[1-9]\d* added, [1-9]\d* changed, [1-9]\d* removed", line) for line in again)
    whole.close()
    streamed.close()


def test_an_id_used_twice_in_a_large_file_keeps_its_first_record(tmp_path, home, large, capsys):
    # A reader that holds no more than a batch can't know the ids before it: the store, which holds their paths, does.
    file = tmp_path / "c" / "big.jsonl"
    write(file, jsonl({"id": n % 10, "text": f"record {n}"} for n in range(25)))  # each id again past its batch
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    assert texts(catalog) == {f"big/{n}": f"record {n}" for n in range(10)}
    twice = "; ".join(f"line {n + 1}: id {n % 10} used twice" for n in range(10, 15))
    assert f"c: big.jsonl: 15 lines skipped ({twice}; ...)" in said(capsys)
    before = doc_ids(catalog)
    write(file, file.read_text(encoding="utf-8") + "\n")  # read again, with every first record as it is indexed
    indexing.index(catalog)
    assert doc_ids(catalog) == before
    told = said(capsys)
    assert f"c: big.jsonl: 15 lines skipped ({twice}; ...)" in told
    assert any("0 added, 0 changed, 0 removed" in line for line in told)


def test_records_gone_from_a_large_file_leave_the_index(tmp_path, home, large, capsys):
    # No batch holds the records that are gone: they are the file's documents that none of its batches named.
    file = tmp_path / "c" / "big.jsonl"
    write(file, numbered(30))
    write(tmp_path / "c" / "other.jsonl", numbered(3))
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    before = doc_ids(catalog)
    kept = [{"id": n, "text": f"record {n}"} for n in range(0, 30, 3)]
    kept[4]["text"] = "changed"
    write(file, jsonl([*kept, {"id": "new", "text": "new"}]))
    capsys.readouterr()
    indexing.index(catalog)
    after = doc_ids(catalog)
    assert sorted(after) == sorted(
        [*(f"big/{n}" for n in range(0, 30, 3)), "big/new", *(f"other/{n}" for n in range(3))]
    )
    assert all(after[f"big/{n}"] == before[f"big/{n}"] for n in range(0, 30, 3) if n != 12)  # the same rows
    assert texts(catalog)["big/12"] == "changed"
    assert any("1 added, 1 changed, 20 removed" in line for line in said(capsys))
    assert Index(catalog).search('"record 29"') == []  # gone from the keyword index too
    assert [h["id"] for h in Index(catalog).search('"record 27"')] == ["c/big/27"]
    assert only(catalog).execute("SELECT count(*) FROM temp.sift_staying").fetchone() == (0,)  # nothing left of the run


def test_storing_a_large_file_again_scans_no_documents(tmp_path, home, large):
    # Each batch's documents are found by their paths, and those gone by their file: found among all the file's, or
    # all the collection's, a batch at a time, a file of millions of records would be read through a thousand times.
    file = tmp_path / "c" / "big.jsonl"
    write(file, numbered(30))
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    write(file, numbered(20))
    db, asked = catalog.store("c"), []
    db.set_trace_callback(lambda statement: statement.startswith("SELECT") and asked.append(statement))
    try:
        indexing.index(catalog)
    finally:
        db.set_trace_callback(None)
    found = [s for s in asked if "FROM docs WHERE path IN" in s or "FROM docs WHERE source = " in s]
    assert len(found) >= 4  # the three batches, and the gone
    for statement in found:
        plan = " ".join(row[-1] for row in db.execute(f"EXPLAIN QUERY PLAN {statement}"))
        assert "SEARCH docs USING" in plan
        assert "SCAN docs" not in plan


@pytest.mark.parametrize("n", [3, 15])  # in the file's first batch, and in a later one
def test_a_record_of_a_large_file_waits_for_the_path_a_file_has(tmp_path, home, large, capsys, n):
    root = tmp_path / "c"
    write(root / "big" / f"r{n}.md", "# File\n")
    catalog = Catalog(home)
    catalog.add(root, "c")
    indexing.index(catalog)
    write(root / "big.jsonl", jsonl({"id": f"r{i}.md", "text": f"record {i}"} for i in range(20)))
    indexing.index(catalog)
    assert len(texts(catalog)) == 20
    assert texts(catalog)[f"big/r{n}.md"] == "# File\n"
    path = f"big/r{n}.md"
    assert f"c: {path}: a document of {path} has this path; this one of big.jsonl is left out until it is free" in said(
        capsys
    )
    (root / "big" / f"r{n}.md").unlink()
    indexing.index(catalog)  # the path is free: its file is read again, in the same run
    assert len(texts(catalog)) == 20
    assert texts(catalog)[f"big/r{n}.md"] == f"record {n}"


def test_a_file_of_a_few_records_grown_large_keeps_those_it_had(tmp_path, home, large, capsys):
    # Its few documents are held as a small file's are, and its new ones looked up as a large file's: the old ones
    # come in a late batch here, one of them twice.
    file = tmp_path / "c" / "big.jsonl"
    write(file, numbered(5))
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    before = doc_ids(catalog)
    grown = [{"id": i, "text": "changed" if i == 2 else f"record {i}"} for i in reversed(range(1, 20))]
    write(file, jsonl([*grown, {"id": 3, "text": "again"}, {"id": 19, "text": "again"}]))
    capsys.readouterr()
    indexing.index(catalog)
    after = doc_ids(catalog)
    assert sorted(after) == sorted(f"big/{i}" for i in range(1, 20))
    assert all(after[f"big/{i}"] == before[f"big/{i}"] for i in (1, 3, 4))  # the same rows
    assert (texts(catalog)["big/2"], texts(catalog)["big/3"], texts(catalog)["big/19"]) == (
        "changed",
        "record 3",
        "record 19",
    )
    told = said(capsys)
    assert "c: big.jsonl: 2 lines skipped (line 20: id 3 used twice; line 21: id 19 used twice)" in told
    assert any("15 added, 1 changed, 1 removed" in line for line in told)


def test_a_large_file_emptied_leaves_the_index(tmp_path, home, large, capsys):
    file = tmp_path / "c" / "big.jsonl"
    write(file, numbered(30))
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    write(file, "")
    capsys.readouterr()
    indexing.index(catalog)
    assert texts(catalog) == {}
    assert any("0 added, 0 changed, 30 removed" in line for line in said(capsys))


def test_a_large_file_the_store_refuses_a_record_of_is_left_out_whole(tmp_path, home, large, monkeypatch, capsys):
    # Stored in one transaction, as a small file is: no batch of it stays without the others.
    root = tmp_path / "c"
    write(root / "big.jsonl", numbered(20))
    write(root / "note.md", "# Note\n")
    catalog = Catalog(home)
    catalog.add(root, "c")
    insert = indexing._insert

    def refusing(db, sid, path, doc):
        if path == "big/15":
            raise sqlite3.DataError("string or blob too big")
        return insert(db, sid, path, doc)

    monkeypatch.setattr(indexing, "_insert", refusing)
    indexing.index(catalog)
    assert sorted(texts(catalog)) == ["note.md"]
    told = said(capsys)
    assert "c: big.jsonl: skipped (string or blob too big)" in told
    assert any("1 added, 0 changed, 0 removed, 1 unreadable" in line for line in told)
    monkeypatch.setattr(indexing, "_insert", insert)
    indexing.index(catalog)  # read again
    assert len(texts(catalog)) == 21


class Torn(io.BytesIO):
    """A file that can be read for its first five hundred bytes, as on a share that stops answering."""

    def read1(self, size=-1):
        if self.tell() > 500:
            raise OSError("the share is gone")
        return super().read1(200)


def test_a_large_file_that_cannot_be_read_to_its_end_is_left_out_whole(tmp_path, home, large, monkeypatch, capsys):
    root = tmp_path / "c"
    write(root / "big.jsonl", numbered(40))
    catalog = Catalog(home)
    catalog.add(root, "c")
    monkeypatch.setattr(readers, "open", lambda path, mode: Torn(Path(path).read_bytes()), raising=False)
    indexing.index(catalog)
    assert texts(catalog) == {}
    told = said(capsys)
    assert "c: big.jsonl: skipped (OSError: the share is gone)" in told
    assert any("0 added, 0 changed, 0 removed, 1 unreadable" in line for line in told)
    monkeypatch.delattr(readers, "open")
    indexing.index(catalog)  # read again
    assert len(texts(catalog)) == 40


def test_a_large_file_starts_no_worker_processes(tmp_path, home, pools, monkeypatch):
    # Workers read files whole, and their documents come back whole: a file read as it is stored is read here, and
    # counted, it started workers that had nothing to read.
    monkeypatch.setattr(readers, "STREAMED", 1000)
    monkeypatch.setattr(indexing, "BYTES_PER_WORKER", 100)
    write(tmp_path / "c" / "big.jsonl", numbered(200))
    write(tmp_path / "c" / "note.md", "# Note\n")
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    assert len(texts(catalog)) == 201
    assert pools == []


def test_reindex_reads_a_large_file_as_it_stores_it(tmp_path, home, large):
    file = tmp_path / "c" / "big.jsonl"
    write(file, numbered(30))
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    before = doc_ids(catalog)
    write(file, jsonl([{"id": 0, "text": "changed"}, *({"id": n, "text": f"record {n}"} for n in range(1, 20)), "x"]))
    (entry, docs), *_ = indexing.reindex(catalog, ["c/big.jsonl"]).files
    assert entry == {"id": "c/big.jsonl", "status": "changed", "note": "1 line skipped (line 21: not an object)"}
    after = doc_ids(catalog)
    assert sorted(docs) == sorted(after.values())
    assert len(after) == 20
    assert all(after[f"big/{n}"] == before[f"big/{n}"] for n in range(1, 20))
    again = indexing.reindex(catalog, ["c/big.jsonl"])
    assert again.files[0][0] == {"id": "c/big.jsonl", "status": "unchanged", "note": entry["note"]}
    assert again.removed == []
    chunks = dict(only(catalog).execute("SELECT d.path, c.id FROM chunks c JOIN docs d ON d.id = c.doc"))
    write(file, numbered(5))
    result = indexing.reindex(catalog, ["c/big.jsonl"])
    assert sorted(result.removed) == sorted(chunks[f"big/{n}"] for n in (0, *range(5, 20)))  # changed, and gone


def test_reindex_leaves_a_large_file_it_cannot_read_to_its_end_as_indexed(tmp_path, home, large, monkeypatch):
    file = tmp_path / "c" / "big.jsonl"
    write(file, numbered(10))
    catalog = Catalog(home)
    catalog.add(tmp_path / "c", "c")
    indexing.index(catalog)
    before = doc_ids(catalog)
    write(file, numbered(40))
    monkeypatch.setattr(readers, "open", lambda path, mode: Torn(Path(path).read_bytes()), raising=False)
    result = indexing.reindex(catalog, ["c/big.jsonl"])
    assert result.files == [({"id": "c/big.jsonl", "status": "unreadable", "note": "OSError: the share is gone"}, [])]
    assert result.removed == []
    assert doc_ids(catalog) == before
