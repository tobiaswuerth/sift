"""Reading files into documents: front matter, titles, chunks with their lines, encodings. Each kind of file has a
reader (see test_documents.py for the contract they all keep, and test_jsonl.py for JSON Lines)."""

import pytest

from sift import readers


def pieces(text, first_line=1):
    """Chunks as (first line, last line, text): chunk() gives spans of the text."""
    return [(first, last, text[pos : pos + length]) for first, last, pos, length in readers.chunk(text, first_line)]


def loaded(path):
    """The one document load() reads from a Markdown or text file, with its chunks as text."""
    (doc,), problems = readers.load(path)
    assert problems == []
    return doc._replace(chunks=[(a, b, doc.text[p : p + n]) for a, b, p, n in doc.chunks])


# --- parsing -----------------------------------------------------------------


def test_parse_front_matter():
    text = '---\ntitle: "A title"\ntags: ["a", "b"]\nyear: 2024\nnote: plain\nq: \'x\'\n  nested: no\n---\n# H\n'
    title, meta, body, first_line = readers.parse(text, "fallback")
    assert title == "A title"
    assert meta == {"title": "A title", "tags": ["a", "b"], "year": 2024, "note": "plain", "q": "x"}
    assert body == "# H\n"
    assert first_line == 9


def test_parse_keeps_json_unsafe_constants_as_text():
    _, meta, _, _ = readers.parse("---\na: NaN\nb: [1, Infinity]\n---\n", "f")
    assert meta == {"a": "NaN", "b": [1, "Infinity"]}


def test_parse_windows_line_endings():
    title, meta, body, first_line = readers.parse("---\r\ntitle: T\r\n---\r\nbody\r\n", "f")
    assert (title, meta, body, first_line) == ("T", {"title": "T"}, "body\r\n", 4)


def test_parse_title_fallbacks():
    assert readers.parse("intro\n## Second level ##\n", "file")[0] == "Second level"
    assert readers.parse("no heading here\n", "file")[0] == "file"
    assert readers.parse('---\ntitle: ""\n---\n# From heading\n', "file")[0] == "From heading"
    assert readers.parse("---\ntitle: 1984\n---\nbody\n", "file")[0] == "1984"  # a number the author chose
    assert readers.parse("---\ntitle: true\n---\nbody\n", "file")[0] == "file"
    assert readers.parse('---\ntitle: ["a", "b"]\n---\nbody\n', "file")[0] == "file"


@pytest.mark.parametrize(
    "value, parsed",
    [
        ("[Hans Mueller, H. Müller]", ["Hans Mueller", "H. Müller"]),  # YAML's flow list, items unquoted
        ('["a", "b"]', ["a", "b"]),
        ("[a, 'b, c', \"d\", O'Brien]", ["a", "b, c", "d", "O'Brien"]),  # a quote starts a quoted item only
        ("[1, two, 3.5, true]", [1, "two", 3.5, True]),
        ("[a, [b, c]]", ["a", ["b", "c"]]),
        ("[a, b, ]", ["a", "b"]),
        ("[ ]", []),
        ("[[Acme AG]]", "[[Acme AG]]"),  # one wikilink: a link, as Obsidian users mean it, not YAML's nested list
        ("[[Acme AG|Acme]]", "[[Acme AG|Acme]]"),
        ("[[[Anna Keller]], [[Peter Frei|Peter]]]", ["[[Anna Keller]]", "[[Peter Frei|Peter]]"]),
        ('["[[Anna Keller]]", "[[Peter Frei]]"]', ["[[Anna Keller]]", "[[Peter Frei]]"]),
        ("[[A]], [[B]]", "[[A]], [[B]]"),  # no list: its first bracket closes early
        ("[Draft] My note", "[Draft] My note"),
        ("[unclosed, list", "[unclosed, list"),
        ('["unclosed, quote]', '["unclosed, quote]'),
        ("[a]]", "[a]]"),
    ],
)
def test_front_matter_flow_lists(value, parsed):
    # `aliases: [Hans Mueller, H. Müller]` was one string, "[Hans Mueller, H. Müller]": no alias matched.
    assert readers.parse(f"---\nkey: {value}\n---\n", "f")[1] == {"key": parsed}


def test_titles_from_file_names():
    # In a vault whose notes start with "## Summary", every note was titled "Summary".
    assert readers.parse("## Summary\n", "Hans Müller", "name")[0] == "Hans Müller"
    assert readers.parse("---\ntitle: Set\n---\n## Summary\n", "Hans Müller", "name")[0] == "Set"
    assert readers.parse("## Summary\n", "Hans Müller", "heading")[0] == "Summary"


def test_load_takes_titles_from_file_names_if_the_collection_says_so(tmp_path):
    path = tmp_path / "Hans Müller.md"
    path.write_text("## Summary\n\nKey account.\n", encoding="utf-8")
    assert readers.load(path, {"title_from": "name"})[0][0].title == "Hans Müller"
    assert readers.load(path)[0][0].title == "Summary"


def test_parse_without_closing_front_matter_is_plain_text():
    title, meta, body, _ = readers.parse("---\ntitle: x\nno end\n", "file")
    assert (title, meta) == ("file", {})
    assert body.startswith("---")


# --- chunking ----------------------------------------------------------------


def test_chunk_short_text_is_one_chunk():
    assert list(pieces("# T\n\nshort body\n", 5)) == [(5, 7, "# T\n\nshort body")]


def test_chunk_lines_are_those_of_the_text_not_of_blank_lines_around_it():
    # the blank line after front matter was reported as a chunk's first line, the blank lines at the end as its last
    assert list(pieces("\n\n# T\n\nbody\n\n\n", 5)) == [(7, 9, "# T\n\nbody")]


@pytest.mark.parametrize("text", ["", "\n\n  \n"])
def test_chunk_empty_text(text):
    assert list(pieces(text)) == []


def test_chunk_splits_with_line_numbers_and_overlap():
    text = "".join(f"line {i} " + "x" * 90 + "\n" for i in range(100))
    chunks = list(pieces(text))
    assert len(chunks) > 1
    assert chunks[0][0] == 1
    assert chunks[-1][1] == 100
    assert all(len(c[2]) <= readers.CHUNK_CHARS for c in chunks)
    assert chunks[1][0] <= chunks[0][1]  # consecutive chunks overlap


def test_chunk_prefers_headings_without_overlap():
    text = "# One\n" + "a " * 900 + "\n# Two\nsecond section\n"
    chunks = list(pieces(text))
    assert [c[2].splitlines()[0] for c in chunks] == ["# One", "# Two"]
    assert chunks[1][0] == chunks[0][1] + 1


def test_chunk_never_exceeds_limit_with_long_lines():
    text = "short line\n" * 20 + "y" * (readers.CHUNK_CHARS - 50) + "\n" + "z " * 3000
    assert all(len(c[2]) <= readers.CHUNK_CHARS for c in pieces(text))


def test_chunk_breaks_unbroken_text():
    chunks = list(pieces("x" * (readers.CHUNK_CHARS * 2 + 10)))
    assert [len(c[2]) for c in chunks] == [readers.CHUNK_CHARS, readers.CHUNK_CHARS, 10]


# --- files -------------------------------------------------------------------


def test_load(tmp_path):
    path = tmp_path / "note.md"
    path.write_bytes("﻿---\ntitle: BOM\n---\nbody \xe9\n".encode())  # a UTF-8 byte order mark first
    doc = loaded(path)
    assert (doc.key, doc.title, doc.meta, doc.chunks) == ("", "BOM", {"title": "BOM"}, [(4, 4, "body \xe9")])
    assert doc.text == "---\ntitle: BOM\n---\nbody \xe9\n"  # the whole file, as get returns it
    assert len(doc.digest) == 32
    assert readers.load(tmp_path / "missing.md") is None


def test_load_normalises_windows_line_endings(tmp_path):
    path = tmp_path / "crlf.md"
    path.write_bytes(b"---\r\ntitle: T\r\n---\r\n# T\r\n\r\nbody\r\n")
    assert loaded(path).chunks == [(4, 6, "# T\n\nbody")]


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
    assert readers.parse(text, "file")[0] == title


def test_heading_search_stays_fast_on_long_lines():
    # A heading followed by a long run of spaces made the heading regex backtrack for minutes, stalling `sift index`.
    text = "# a" + " " * 100_000 + "b\n" + "#" + " " * 100_000 + "\n"
    assert readers.parse(text, "file")[0] == "a" + " " * 100_000 + "b"


def test_parse_empty_front_matter():
    assert readers.parse("---\n---\n# Title\nbody\n", "file") == ("Title", {}, "# Title\nbody\n", 3)


def test_parse_front_matter_lists_and_comments():
    # Obsidian, Jekyll and Hugo write lists as one `- item` per line.
    text = '---\n# a comment\ntags:\n  - painting\n  - "robots"\n- 2024\nauthors: ["A"]\nempty:\n---\nbody\n'
    assert readers.parse(text, "file")[1] == {"tags": ["painting", "robots", 2024], "authors": ["A"], "empty": None}


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
