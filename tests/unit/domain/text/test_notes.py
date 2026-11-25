import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.api.collections import TitleFrom
from sift.domain.text.decoding import text_pieces
from sift.domain.text.notes import Document, digest, document_of, file_stem, note, read_note


def written(text: str, name: str = "file.md", title_from: TitleFrom = TitleFrom.HEADING) -> Document:
    return note(text.encode(), name, title_from)


def passages(document: Document) -> list[tuple[int, int, str]]:
    return [(c.first_line, c.last_line, document.text[c.start : c.start + c.length]) for c in document.chunks]


def test_a_note_s_passages_carry_the_line_numbers_of_the_file() -> None:
    text = '---\ntitle: "A title"\ntags: ["a", "b"]\nyear: 2024\nnote: plain\nq: \'x\'\n  nested: no\n---\n# H\n'
    document = written(text)
    assert (document.key, document.title, passages(document)) == ("", "A title", [(9, 9, "# H")])
    assert document.text == text


def test_a_title_of_several_lines_is_one_line_and_the_field_stays_as_written() -> None:
    assert written("---\ntitle: |\n  Annual report\n  2026\n---\nbody\n").title == "Annual report 2026"
    document = written('---\ntitle: "two \\n lines"\n---\n')
    assert (document.title, document.fields) == ("two lines", {"title": "two \n lines"})
    assert written("---\ntitle: Two  spaces\n---\n").title == "Two  spaces"


@pytest.mark.parametrize(
    "text, title",
    [
        ("intro\n## Second level ##\n", "Second level"),
        ("no heading here\n", "file"),
        ('---\ntitle: ""\n---\n# From heading\n', "From heading"),
        ("---\ntitle: 1984\n---\nbody\n", "1984"),
        ("---\ntitle: true\n---\nbody\n", "file"),
        ('---\ntitle: ["a", "b"]\n---\nbody\n', "file"),
        ("# Learning C#\n", "Learning C#"),
        ("#\nA paragraph, not a heading\n# Real\n", "Real"),
        ("# ##\n# Real\n", "Real"),
        ("   # Indented up to three spaces\n", "Indented up to three spaces"),
        ("    # Four spaces make it code\n", "file"),
        ("#hashtag\n", "file"),
        ("Intro\n\n```bash\n# install it\n```\n# After the code\n", "After the code"),
        ("~~~\n# not a heading\n~~~\n", "file"),
        ("````\n```\n# still code\n````\n", "file"),
        ("```\n# unclosed code block\n", "file"),
        ("---\n---\n# Title\nbody\n", "Title"),
        ("```inline``` is code\n\n# Real heading\n", "Real heading"),
    ],
)
def test_the_title_is_the_title_field_then_the_first_heading_then_the_file_name(text: str, title: str) -> None:
    assert written(text).title == title


def test_titles_may_come_from_file_names_after_the_title_field() -> None:
    assert written("## Summary\n", "Hans Müller.md", TitleFrom.NAME).title == "Hans Müller"
    assert written("---\ntitle: Set\n---\n## Summary\n", "Hans Müller.md", TitleFrom.NAME).title == "Set"
    assert written("## Summary\n", "Hans Müller.md").title == "Summary"


def test_a_heading_is_found_on_lines_that_make_a_careless_pattern_backtrack() -> None:
    text = "# a" + " " * 100_000 + "b\n" + "#" + " " * 100_000 + "\n"
    assert written(text).title == "a" + " " * 100_000 + "b"


@pytest.mark.parametrize(
    "name, stem",
    [("Note.md", "Note"), ("a.tar.gz", "a.tar"), (".md", ".md"), ("name.", "name"), ("plain", "plain"), ("..", "..")],
)
def test_a_file_s_stem_is_its_name_without_its_last_extension_on_every_python(name: str, stem: str) -> None:
    assert file_stem(name) == stem
    assert written("no heading\n", name).title == stem


def test_a_note_of_front_matter_alone_is_its_front_matter() -> None:
    document = written('---\norg: "[[Acme AG]]"\nemail: peter@acme.example\n---\n\n', "Peter Frei.md")
    assert passages(document) == [(2, 3, 'org: "[[Acme AG]]"\nemail: peter@acme.example')]
    assert written("---\n---\n", "Empty.md").chunks == ()
    assert written("---\n  \n---\n", "Blank.md").chunks == ()
    assert len(written("---\na: 1\n---\nA body.\n", "Body.md").chunks) == 1


@pytest.mark.parametrize(
    "text", ["---\nSome intro words xyzzy\n---\nbody text\n", "---\ntitle: x\nxyzzy, a line of prose\n---\nbody\n"]
)
def test_text_between_two_rules_that_is_no_front_matter_stays_searchable(text: str) -> None:
    document = written(text, "Rules.md")
    assert (document.title, document.fields) == ("Rules", {})
    assert any("xyzzy" in passage for _, _, passage in passages(document))


def test_a_byte_order_mark_is_left_out_and_the_digest_is_of_the_bytes() -> None:
    raw = "\N{ZERO WIDTH NO-BREAK SPACE}---\ntitle: BOM\n---\nbody \xe9\n".encode()
    document = note(raw, "note.md")
    assert (document.title, document.fields, passages(document)) == ("BOM", {"title": "BOM"}, [(4, 4, "body \xe9")])
    assert document.text == "---\ntitle: BOM\n---\nbody \xe9\n"
    assert len(document.digest) == 32
    assert document.digest != note(raw + b" ", "note.md").digest


def test_windows_line_endings_are_newlines() -> None:
    document = note(b"---\r\ntitle: T\r\n---\r\n# T\r\n\r\nbody\r\n", "crlf.md")
    assert passages(document) == [(4, 6, "# T\n\nbody")]


@pytest.mark.parametrize(
    "mark, codec",
    [
        (b"\xff\xfe", "utf-16-le"),
        (b"\xfe\xff", "utf-16-be"),
        (b"\xff\xfe\x00\x00", "utf-32-le"),
        (b"\x00\x00\xfe\xff", "utf-32-be"),
    ],
)
def test_utf16_and_utf32_after_their_byte_order_marks(mark: bytes, codec: str) -> None:
    document = note(mark + "# Héllo\r\nworld\r\n".encode(codec), "note.txt")
    assert (document.title, document.fields, passages(document)) == ("Héllo", {}, [(1, 2, "# Héllo\nworld")])


def test_what_is_no_utf8_reads_as_windows_ansi() -> None:
    raw = "Grüezi Müller, 5 €\n".encode("cp1252") + "naïve \N{CHECK MARK}\n".encode() + b"r\xe9sum\xe9 \x81\n"
    expected = "Grüezi Müller, 5 €\nnaïve \N{CHECK MARK}\nrésumé \N{REPLACEMENT CHARACTER}"
    assert passages(note(raw, "ansi.txt")) == [(1, 3, expected)]


def test_only_markdown_files_have_links() -> None:
    assert written("[[A]]\n", "notes.txt").links == ()
    assert written("[[A]]\n", "notes.markdown").links != ()
    document = note(b"---\r\norg: '[[Acme]]'\r\n---\r\n\r\nSee [[B]].\r\n", "Note.MD")
    assert [(link.field, link.line, link.target) for link in document.links] == [("org", 2, "Acme"), (None, 5, "B")]


def test_a_nul_is_kept_as_a_replacement_character_of_the_same_length() -> None:
    document = note(b"# Num\x00bers\n\nwombat\x00\x00 numbat\n", "pasted.md")
    replaced = note(b"# Num\xef\xbf\xbdbers\n\nwombat\xef\xbf\xbd\xef\xbf\xbd numbat\n", "pasted.md")
    assert (document.title, document.text) == (replaced.title, replaced.text)
    assert document.title == "Num\N{REPLACEMENT CHARACTER}bers"
    assert document.chunks == replaced.chunks


MARKED = st.sampled_from(["a", "#", " ", "\n", "\r", "-", ":", "[", "]", "(", ")", "`", "~", ">", "|", "\N{NULL}", "é"])
notes_written = st.one_of(
    st.text(MARKED, max_size=300),
    st.tuples(st.text(MARKED, max_size=80), st.text(MARKED, max_size=300)).map(
        lambda both: f"---\n{both[0]}\n---\n{both[1]}"
    ),
).map(str.encode)


@given(notes_written, st.lists(st.integers(0, 400), max_size=5), st.sampled_from(["n.md", "n.txt"]))
def test_a_note_read_in_pieces_is_the_note_read_whole(raw: bytes, at: list[int], name: str) -> None:
    edges = [0, *sorted(min(each, len(raw)) for each in at), len(raw)]
    parts = [raw[start:end] for start, end in itertools.pairwise(edges)]
    assert document_of(read_note(text_pieces(parts), name), digest(raw)) == note(raw, name)


NESTED = b"---\nauthor:\n  name: Zanzibarian\npeople:\n  - name: A\n    role: Wombatchief\n---\nbody\n"


def test_words_of_nested_front_matter_are_in_some_passage() -> None:
    read = " ".join(text for _, _, text in passages(note(NESTED, "n.md")))
    assert ("Zanzibarian" in read, "Wombatchief" in read, "body" in read) == (True, True, True)


def test_front_matter_read_whole_is_no_passage_of_its_own_beside_a_body() -> None:
    found = note(b"---\nauthor: Zanzibarian\ntags:\n  - a\n---\nbody\n", "n.md")
    assert [text.strip() for _, _, text in passages(found)] == ["body"]
