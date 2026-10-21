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


def test_a_note_of_front_matter_alone_is_its_front_matter():
    # A vault's contact notes often hold properties only: without a passage, no search could find them, not even by
    # their name.
    doc = readers.note('---\norg: "[[Acme AG]]"\nemail: peter@acme.example\n---\n\n', "Peter Frei.md")
    assert [(a, b, doc.text[p : p + n]) for a, b, p, n in doc.chunks] == [
        (2, 3, 'org: "[[Acme AG]]"\nemail: peter@acme.example')
    ]
    assert readers.note("---\n---\n", "Empty.md").chunks == []
    assert readers.note("---\n  \n---\n", "Blank.md").chunks == []
    assert len(readers.note("---\na: 1\n---\nA body.\n", "Body.md").chunks) == 1  # the body alone, as before


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


# --- links -------------------------------------------------------------------


def links(text, name="Note.md"):
    """The links readers find in a note, as (field, line, kind, target, fragment, shown)."""
    return [(lk.field, lk.line, lk.kind, lk.target, lk.fragment, lk.shown) for lk in readers.note(text, name).links]


@pytest.mark.parametrize(
    "written, link",
    [
        ("[[Hans Müller]]", ("wikilink", "Hans Müller", None, None)),
        ("[[Hans Müller.md]]", ("wikilink", "Hans Müller.md", None, None)),
        ("[[01_People/Hans Müller]]", ("wikilink", "01_People/Hans Müller", None, None)),
        ("[[Hans Müller|Hans]]", ("wikilink", "Hans Müller", None, "Hans")),
        ("| [[Hans Müller\\|Hans]] |", ("wikilink", "Hans Müller", None, "Hans")),  # in a table, | is escaped
        ("[[Hans Müller#Kontakt]]", ("wikilink", "Hans Müller", "Kontakt", None)),
        ("[[Hans Müller#^ab12cd|Hans]]", ("wikilink", "Hans Müller", "^ab12cd", "Hans")),
        ("[[ Hans Müller ]]", ("wikilink", "Hans Müller", None, None)),
        ("[[Release v1.2]]", ("wikilink", "Release v1.2", None, None)),
        ("![[contract.pdf]]", ("embed", "contract.pdf", None, None)),
        ("![[Hans Müller#Kontakt]]", ("embed", "Hans Müller", "Kontakt", None)),
        ("[text](Other%20Note.md)", ("markdown", "Other Note.md", None, "text")),
        ("[text](<Other Note.md>)", ("markdown", "Other Note.md", None, "text")),
        ("[text](../People/Hans.md#Kontakt)", ("markdown", "../People/Hans.md", "Kontakt", "text")),
        ('[text](Note.md "A title")', ("markdown", "Note.md", None, "text")),
        ("[see [1]](Note%20(old).md)", ("markdown", "Note (old).md", None, "see [1]")),
        ("![scan](scan.pdf)", ("markdown", "scan.pdf", None, "scan")),
        ("[](Note.md)", ("markdown", "Note.md", None, None)),
    ],
)
def test_links_in_a_note(written, link):
    assert links(f"Before {written} after.\n") == [(None, 1, *link)]


@pytest.mark.parametrize(
    "written",
    [
        "[[#Kontakt]]",  # into this note
        "[[]]",
        "[[ |shown]]",
        "[[Folder/]]",
        "![[diagram.png]]",  # sift doesn't index images: neither found nor missing
        "![alt](img/diagram.PNG)",
        "[text](https://example.com/a.md)",
        "[text](mailto:hans@example.com)",
        "[text](obsidian://open?vault=x)",
        "[text](#heading)",
        "[text](//example.com/a.md)",
        "[text](..)",
        "[two words](Some Note.md)",  # not a Markdown link: a space ends the address
        "`[[Not a link]]`",
        "``a ` [[Not a link]] b``",
    ],
)
def test_what_is_no_link(written):
    assert links(f"{written}\n") == []


def test_links_in_code_blocks_are_no_links():
    text = "```\n[[Also not a link]]\n```\n~~~python\n[x](y.md)\n```\n[[still code]]\n~~~\n[[Real]]\n"
    assert links(text) == [(None, 9, "wikilink", "Real", None, None)]


def test_an_unmatched_backtick_hides_nothing():
    assert links("It's 5` long, see [[Real]] and ``[[code]]``.\n") == [(None, 1, "wikilink", "Real", None, None)]


def test_several_links_on_a_line_in_order():
    found = links("[b](B.md) and [[A]], ![[C.pdf]] and [[D]] [d](D.md)\n")
    assert [(kind, target) for _, _, kind, target, _, _ in found] == [
        ("markdown", "B.md"),
        ("wikilink", "A"),
        ("embed", "C.pdf"),
        ("wikilink", "D"),
        ("markdown", "D.md"),
    ]


def test_a_wikilink_in_a_markdown_link_counts_once():
    assert links("[[A]](B.md)\n") == [(None, 1, "wikilink", "A", None, None)]


def test_links_in_front_matter():
    text = (
        "---\n"
        'org: "[[Acme AG]]"\n'
        "reports_to: [[Peter Frei]]\n"
        'colleague_of: ["[[Anna Keller]]", "[[Peter Frei|Peter]]"]\n'
        "team:\n"
        '  - "[[Hans Müller]]"\n'
        "  - not a link\n"
        "# also: [[Commented out]]\n"
        "contacts:\n"
        '  primary: "[[Hans Müller#Kontakt]]"\n'
        "url: https://example.com/[x](y.md)\n"
        "---\n"
        "Body with [[Acme AG]].\n"
    )
    assert links(text) == [
        ("org", 2, "wikilink", "Acme AG", None, None),
        ("reports_to", 3, "wikilink", "Peter Frei", None, None),
        ("colleague_of", 4, "wikilink", "Anna Keller", None, None),
        ("colleague_of", 4, "wikilink", "Peter Frei", None, "Peter"),
        ("team", 6, "wikilink", "Hans Müller", None, None),
        ("contacts", 10, "wikilink", "Hans Müller", "Kontakt", None),
        (None, 13, "wikilink", "Acme AG", None, None),
    ]


def test_link_keys_compare_names_as_obsidian_does():
    # Case-insensitively, with the extension of a note left out, and composed: file names arrive decomposed from macOS.
    decomposed = "Hans Mu\u0308ller"
    text = f"[[{decomposed}]] [[hans müller.md]] [[01_People/HANS MÜLLER#x]]"
    keys = [lk.key for lk in readers.note(text, "N.md").links]
    assert keys == ["hans müller"] * 3
    assert [lk.key for lk in readers.note("![[Contract.PDF]] [[St.Gallen]]", "N.md").links] == [
        "contract.pdf",
        "st.gallen",
    ]


def test_only_markdown_files_have_links(tmp_path):
    assert readers.note("[[A]]\n", "notes.txt").links == ()
    assert readers.note("[[A]]\n", "notes.markdown").links != ()
    path = tmp_path / "Note.md"
    path.write_bytes(b"---\r\norg: '[[Acme]]'\r\n---\r\n\r\nSee [[B]].\r\n")
    (doc,), _ = readers.load(path)
    assert [(lk.field, lk.line, lk.target) for lk in doc.links] == [("org", 2, "Acme"), (None, 5, "B")]


def test_link_search_stays_fast_on_long_lines():
    # Every line of a note is searched for links; a regex that backtracks would stall `sift index` on one long line.
    for line in ("[" * 200_000, "[[" * 100_000, "[a](" * 50_000, "`" * 100_000 + "x", "[x] " * 50_000 + "](y"):
        assert readers.note(line + "\n" + "](\n", "Note.md").links == ()
