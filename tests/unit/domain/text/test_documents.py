from sift.domain.text.documents import NO_TEXT, Converted, Scan, Unread, document
from sift.domain.text.notes import Document, digest


def converted(markdown: str, **properties: str | list[str]) -> Document:
    found = document(b"raw", "pigments.pdf", Converted(markdown, properties))
    assert isinstance(found, Document)
    return found


def test_a_document_is_its_markdown_with_its_properties_as_fields() -> None:
    found = converted("# Heading\n\nwords\n", title="Pigment mixing", authors=["Ada Example"])
    assert (found.key, found.title, found.text) == ("", "Pigment mixing", "# Heading\n\nwords\n")
    assert found.fields == {"title": "Pigment mixing", "authors": ["Ada Example"]}
    assert (found.digest, found.links) == (digest(b"raw"), ())
    assert [found.text[chunk.start : chunk.start + chunk.length] for chunk in found.chunks] == ["# Heading\n\nwords"]


def test_the_title_is_the_title_property_on_one_line_then_the_first_heading_then_the_file_name() -> None:
    assert converted("# Heading\n", title="Two\r\n lines").title == "Two lines"
    assert converted("text\n\n# Heading\n").title == "Heading"
    assert converted("text\n").title == "pigments"


def test_properties_are_composed_and_a_null_character_is_stored_as_a_replacement() -> None:
    found = converted("a\N{NULL}b", subject="Zu\N{COMBINING DIAERESIS}rich")
    assert (found.text, found.fields) == (
        "a\N{REPLACEMENT CHARACTER}b",
        {"subject": "Z\N{LATIN SMALL LETTER U WITH DIAERESIS}rich"},
    )


def test_a_document_without_text_or_not_converted_is_unread_but_a_pdf_without_text_is_a_scan() -> None:
    assert document(b"raw", "notes.docx", Converted(" \n\n")) == Unread(NO_TEXT)
    assert document(b"raw", "broken.pdf", Unread("Parsing error")) == Unread("Parsing error")
    found = document(b"raw", "Scan.PDF", Converted(" \n\n", {"title": "Delivery"}))
    assert isinstance(found, Scan)
    assert (found.raw, found.document.title, found.document.text, found.document.chunks) == (b"raw", "Delivery", "", ())
    untitled = document(b"raw", "scan.pdf", Converted(""))
    assert isinstance(untitled, Scan)
    assert (untitled.document.title, untitled.document.digest) == ("scan", digest(b"raw"))
