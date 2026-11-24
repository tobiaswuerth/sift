import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from xberg import ExtractionResult

from sift.domain.text.documents import FileRead, Unread
from sift.domain.text.kinds import DOCUMENTS
from sift.ingest import converter
from sift.ingest.converter import converted
from tests.mails import PARAGRAPH, SAMPLES, only, read, sample

if TYPE_CHECKING:
    from sift.ingest.documents import Converter

PAGES = (".html", ".htm")
WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _line(text: str, at: int) -> int:
    return text.count("\n", 0, at) + 1


def test_every_kind_of_document_has_a_sample() -> None:
    assert {path.suffix for path in SAMPLES.glob("sample.*")} == {*DOCUMENTS, ".eml"}


@pytest.mark.parametrize("extension", DOCUMENTS)
def test_every_document_reads_whole_as_markdown_in_chunks_of_its_lines(extension: str) -> None:
    raw = sample(extension).read_bytes()
    found = only(raw, f"sample{extension}")
    assert f"This sample is the {extension[1:].upper()} file." in found.text
    assert found.title.strip()
    assert json.loads(json.dumps(found.fields)) == found.fields
    for chunk in found.chunks:
        piece = found.text[chunk.start : chunk.start + chunk.length]
        assert piece == piece.strip() != ""
        assert (chunk.first_line, chunk.last_line) == (
            _line(found.text, chunk.start),
            _line(found.text, chunk.start + chunk.length - 1),
        )
    assert any(PARAGRAPH in found.text[chunk.start : chunk.start + chunk.length] for chunk in found.chunks)
    assert only(raw, f"sample{extension}") == found


@pytest.mark.parametrize("extension", sorted(set(DOCUMENTS) - set(PAGES)))
def test_a_document_s_properties_are_its_fields(extension: str) -> None:
    found = only(sample(extension).read_bytes(), f"sample{extension}")
    assert (found.title, found.fields["title"], found.fields["authors"]) == (
        "Pigment mixing",
        "Pigment mixing",
        ["Ada Example"],
    )
    created = found.fields["created"]
    assert isinstance(created, str)
    assert created.startswith("2024-05-01")


def test_a_pdf_has_its_properties_and_large_bold_text_as_a_heading() -> None:
    found = only(sample(".pdf").read_bytes(), "sample.pdf")
    assert found.fields == {
        "title": "Pigment mixing",
        "authors": ["Ada Example"],
        "keywords": ["paint", "colour"],
        "created": "2024-05-01T12:00:00Z",
    }
    assert found.text.startswith("# Kubelka-Munk theory\n\n")


def test_office_files_become_markdown() -> None:
    assert only(sample(".docx").read_bytes(), "a.docx").text == (
        f"# Kubelka-Munk theory\n\n{PARAGRAPH}\n\nThis sample is the DOCX file.\n"
    )
    assert "## About this sample\n\nThis sample is the PPTX file." in only(sample(".pptx").read_bytes(), "a.pptx").text
    sheet = only(sample(".xlsx").read_bytes(), "a.xlsx").text
    assert sheet.startswith("## Kubelka-Munk theory\n")
    assert "| Ultramarine | PB29 | This sample is the XLSX file. |" in sheet


@pytest.mark.parametrize("extension", PAGES)
def test_a_web_page_is_its_text_titled_by_its_heading(extension: str) -> None:
    found = only(sample(extension).read_bytes(), f"page{extension}")
    kind = extension[1:].upper()
    assert found.text == f"# Kubelka-Munk theory\n\n{PARAGRAPH}\n\nThis sample is the {kind} file.\n"
    assert found.title == "Kubelka-Munk theory"


def test_a_name_s_extension_in_any_case_reads_the_same() -> None:
    raw = sample(".pdf").read_bytes()
    assert only(raw, "SCAN.PDF").text == only(raw, "scan.pdf").text


@pytest.mark.parametrize(
    "name, raw", [("broken.pdf", b"%PDF-1.4 the rest is missing"), ("broken.docx", b"PK\x03\x04"), ("empty.odt", b"")]
)
def test_a_damaged_document_is_unread_with_what_the_converter_said(name: str, raw: bytes) -> None:
    found = read(raw, name)
    assert isinstance(found, Unread)
    assert found.problem.startswith("Parsing error: ")


def test_a_pdf_without_text_is_a_scan_to_read_with_ocr() -> None:
    raw = (SAMPLES / "blank.pdf").read_bytes()
    found = read(raw, "blank.pdf")
    assert isinstance(found, FileRead)
    (scan,) = found.scans
    assert (found.documents, scan.raw, scan.document.title) == ((), raw, "blank")


def _rezipped(path: Path, document: bytes | None = None) -> Path:
    with zipfile.ZipFile(sample(".docx")) as source, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as made:
        for item in source.infolist():
            made.writestr(
                item.filename, document if document and item.filename == "word/document.xml" else source.read(item)
            )
    return path


def _word(body: str) -> bytes:
    return f'<?xml version="1.0"?><w:document xmlns:w="{WORD}"><w:body>{body}</w:body></w:document>'.encode()


def test_an_office_file_that_compresses_as_well_as_deflate_can_is_no_zip_bomb(tmp_path: Path) -> None:
    body = "".join(f"<w:p><w:r><w:t>{'x' * 100_000}</w:t></w:r></w:p>" for _ in range(20))
    path = _rezipped(tmp_path / "repetitive.docx", _word(body))
    with zipfile.ZipFile(path) as made:
        entry = made.getinfo("word/document.xml")
        assert entry.file_size / entry.compress_size > 800
    assert only(path.read_bytes(), path.name).text.count("x") == 2_000_000


def test_a_zip_bomb_is_unread(tmp_path: Path) -> None:
    path = tmp_path / "bomb.docx"
    unpacked = 110 << 20
    with (
        zipfile.ZipFile(sample(".docx")) as source,
        zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as made,
    ):
        for item in source.infolist():
            if item.filename != "word/document.xml":
                made.writestr(item.filename, source.read(item))
                continue
            with made.open(item.filename, "w", force_zip64=True) as written:
                written.write(_word("<w:p><w:r><w:t>").removesuffix(b"</w:body></w:document>"))
                written.write(b"x" * unpacked)
                written.write(b"</w:t></w:r></w:p></w:body></w:document>")
    found = read(path.read_bytes(), path.name)
    assert isinstance(found, Unread)
    assert "exceeds limit" in found.problem


class PanicError(BaseException):
    pass


def test_a_panic_of_the_converter_leaves_the_document_unread_and_an_interrupt_stops_it() -> None:
    def panicking(_raw: bytes, _name: str) -> ExtractionResult:
        raise PanicError("index out of bounds")

    def interrupted(_raw: bytes, _name: str) -> ExtractionResult:
        raise KeyboardInterrupt

    def mistaken(_raw: bytes, _name: str) -> ExtractionResult:
        raise TypeError("a bug")

    def too_deep(_raw: bytes, _name: str) -> ExtractionResult:
        raise RecursionError("maximum recursion depth exceeded")

    assert converted(b"%PDF", "a.pdf", panicking) == Unread("the converter failed: index out of bounds")
    with pytest.raises(KeyboardInterrupt):
        converted(b"%PDF", "a.pdf", interrupted)
    with pytest.raises(TypeError, match="a bug"):
        converted(b"%PDF", "a.pdf", mistaken)
    with pytest.raises(RecursionError):
        converted(b"%PDF", "a.pdf", too_deep)


def test_the_converter_loaded_late_is_what_its_callers_take_it_for() -> None:
    late: Converter = converter
    assert late.converted is converted
