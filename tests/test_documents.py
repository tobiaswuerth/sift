"""Reading files by extension: one reader per kind of file, and PDF, office and other documents converted to Markdown.

Every extension sift reads has a sample in tests/samples, all with the same content (see make_samples.py), so one
contract covers every reader."""

import json
import re
import shutil
import zipfile
from pathlib import Path

import pytest

from sift import store
from sift.search import Index

from .conftest import index_contents, write

SAMPLES = Path(__file__).parent / "samples"
PARAGRAPH = "Kubelka-Munk theory predicts the colour of mixed paints from how each pigment absorbs and scatters light."
PAGES = (".html", ".htm")
WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def sample(ext):
    return SAMPLES / f"sample{ext}"


def read(ext):
    """The one document of a sample."""
    (doc,), problems = store.load(sample(ext))
    assert problems == []
    return doc


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


# --- the contract every reader keeps -----------------------------------------


def test_every_extension_sift_reads_has_a_sample():
    assert {p.suffix for p in SAMPLES.glob("sample.*") if p.suffix != ".py"} == set(store.READERS)


@pytest.mark.parametrize("ext", sorted(store.READERS))
def test_every_reader_keeps_the_contract(ext):
    docs, problems = store.load(sample(ext))
    assert problems == []
    (doc,) = docs
    assert f"This sample is the {ext[1:].upper()} file." in doc.text  # its own content, read in full
    assert isinstance(doc.title, str)
    assert doc.title.strip()
    assert doc.digest
    assert json.loads(json.dumps(doc.meta)) == doc.meta  # metadata is JSON, as the index stores it
    assert doc.chunks
    for first, last, pos, length in doc.chunks:  # chunks are spans of the text, with the lines they cover
        piece = doc.text[pos : pos + length]
        assert piece
        assert piece == piece.strip()
        assert (first, last) == (line_of(doc.text, pos), line_of(doc.text, pos + length - 1))
    assert any(PARAGRAPH in doc.text[pos : pos + length] for _, _, pos, length in doc.chunks)
    assert store.load(sample(ext)) == (docs, problems)  # the same file reads the same every time


def test_extensions_match_in_any_case(tmp_path):
    shutil.copy(sample(".pdf"), tmp_path / "SCAN.PDF")
    shutil.copy(sample(".md"), tmp_path / "Notes.MD")
    assert store.load(tmp_path / "SCAN.PDF") == store.load(sample(".pdf"))
    assert store.load(tmp_path / "Notes.MD")[0][0].text == read(".md").text
    assert sorted(Path(p).name for p, _, _ in store.walk(str(tmp_path))) == ["Notes.MD", "SCAN.PDF"]


def test_a_new_kind_of_file_needs_only_a_reader(tmp_path, home, monkeypatch):
    def read_csv(raw, name, fields):
        text = raw.decode()
        return [store.Doc("", "digest", name, {}, text, list(store.chunk(text)))], []

    monkeypatch.setitem(store.READERS, ".csv", read_csv)
    write(tmp_path / "c" / "pigments.csv", "pigment,index\nUltramarine,PB29\n")
    db = store.connect(home)
    store.add_collection(db, tmp_path / "c", "c")
    store.index(db)
    assert Index(db).get("c/pigments.csv") == "pigment,index\nUltramarine,PB29\n"


# --- documents: PDF, office files, e-books and web pages --------------------------


@pytest.mark.parametrize("ext", sorted(set(store.DOCUMENTS) - set(PAGES)))
def test_document_properties_are_metadata(ext):
    doc = read(ext)
    assert doc.title == "Pigment mixing"  # the document's own title, not its first heading
    assert (doc.meta["title"], doc.meta["authors"]) == ("Pigment mixing", ["Ada Example"])
    assert doc.meta["created"].startswith("2024-05-01")


def test_a_pdf_has_its_properties_as_metadata():
    doc = read(".pdf")
    assert doc.meta == {
        "title": "Pigment mixing",
        "authors": ["Ada Example"],
        "keywords": ["paint", "colour"],
        "created": "2024-05-01T12:00:00Z",
    }
    assert doc.text.startswith("# Kubelka-Munk theory\n\n")  # large bold text becomes a heading


def test_documents_become_markdown():
    assert read(".docx").text == f"# Kubelka-Munk theory\n\n{PARAGRAPH}\n\nThis sample is the DOCX file.\n"
    assert "## About this sample\n\nThis sample is the PPTX file." in read(".pptx").text  # a slide's title
    assert "| Ultramarine | PB29 | This sample is the XLSX file. |" in read(".xlsx").text  # a sheet is a table
    assert read(".xlsx").text.startswith("## Kubelka-Munk theory\n")  # the sheet's name


@pytest.mark.parametrize("ext", PAGES)
def test_a_web_page_is_its_text(ext):
    doc = read(ext)
    assert doc.text == f"# Kubelka-Munk theory\n\n{PARAGRAPH}\n\nThis sample is the {ext[1:].upper()} file.\n"
    assert doc.title == "Kubelka-Munk theory"


@pytest.mark.parametrize(
    "name, raw",
    [("broken.pdf", b"%PDF-1.4 the rest is missing"), ("broken.docx", b"PK\x03\x04"), ("empty.odt", b"")],
)
def test_a_damaged_document_is_skipped_with_the_reason(tmp_path, name, raw):
    (tmp_path / name).write_bytes(raw)
    docs, (problem,) = store.load(tmp_path / name)
    assert docs == []
    assert re.fullmatch(r"skipped \(.+\)", problem)


def test_an_office_file_that_compresses_very_well_is_no_zip_bomb(tmp_path):
    # Office files are zipped XML, and slides or sheets full of repeated markup compress at up to about 1000:1, the
    # most deflate can do. Xberg's default limit of 100:1 took such files for zip bombs and skipped them.
    body = "".join(f"<w:p><w:r><w:t>{'x' * 100_000}</w:t></w:r></w:p>" for _ in range(20))
    document = f'<?xml version="1.0"?><w:document xmlns:w="{WORD}"><w:body>{body}</w:body></w:document>'
    path = tmp_path / "repetitive.docx"
    with zipfile.ZipFile(sample(".docx")) as src, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            dst.writestr(item.filename, document if item.filename == "word/document.xml" else src.read(item))
    with zipfile.ZipFile(path) as z:
        entry = z.getinfo("word/document.xml")
        assert entry.file_size / entry.compress_size > 800  # as compressed as the slides that were skipped
    docs, problems = store.load(path)
    assert problems == []
    assert docs[0].text.count("x") == 2_000_000


def test_a_real_zip_bomb_is_still_skipped(tmp_path):
    path = tmp_path / "bomb.docx"  # a 0.5 MB Word file whose text unpacks to 110 MB
    with (
        zipfile.ZipFile(sample(".docx")) as src,
        zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as dst,
    ):
        for item in src.infolist():
            if item.filename != "word/document.xml":
                dst.writestr(item.filename, src.read(item))
                continue
            with dst.open(item.filename, "w", force_zip64=True) as f:
                f.write(f'<?xml version="1.0"?><w:document xmlns:w="{WORD}"><w:body><w:p><w:r><w:t>'.encode())
                for _ in range(110):
                    f.write(b"x" * (1 << 20))
                f.write(b"</w:t></w:r></w:p></w:body></w:document>")
    docs, (problem,) = store.load(path)
    assert docs == []
    assert "exceeds limit" in problem


def test_a_document_without_text_is_skipped():
    assert store.load(SAMPLES / "blank.pdf") == ([], ["skipped (no text in it: a scan? sift runs no OCR)"])


# --- indexing documents --------------------------------------------------------


@pytest.fixture
def library(tmp_path, home):
    """A collection of one PDF, one Word and one Excel file."""
    root = tmp_path / "library"
    root.mkdir()
    for ext in (".pdf", ".docx", ".xlsx"):
        shutil.copy(sample(ext), root / f"pigments{ext}")
    db = store.connect(home)
    store.add_collection(db, root, "library")
    store.index(db)
    return db, root


def test_documents_are_indexed_and_found(library):
    db, _ = library
    index = Index(db)
    hits = index.search("absorbs scatters", fields=["authors", "created"])
    assert sorted(h["id"] for h in hits) == ["library/pigments.docx", "library/pigments.pdf", "library/pigments.xlsx"]
    assert all(h["fields"]["authors"] == ["Ada Example"] for h in hits)
    assert [h["id"] for h in index.search("Ultramarine")] == ["library/pigments.xlsx"]
    assert index.get("library/pigments.docx") == read(".docx").text  # the Markdown, from the index


def test_a_damaged_document_is_reported_once_and_read_again_when_it_changes(library, capsys):
    db, root = library
    (root / "broken.pdf").write_bytes(b"%PDF-1.4 the rest is missing")
    store.index(db)
    assert "library: broken.pdf: skipped (" in capsys.readouterr().err
    store.index(db)
    assert "broken.pdf" not in capsys.readouterr().err  # unchanged since, so not read again
    shutil.copy(sample(".pdf"), root / "broken.pdf")
    store.index(db)
    assert "1 added" in capsys.readouterr().err
    assert Index(db).get("library/broken.pdf") == read(".pdf").text


def test_documents_are_converted_in_worker_processes(library, tmp_path, monkeypatch, pools):
    db, root = library
    monkeypatch.setattr(store, "_cpus", lambda: 3)  # two workers, one CPU for writing
    monkeypatch.setattr(store, "FILES_PER_WORKER", 1)
    parallel = store.connect(tmp_path / "parallel")
    store.add_collection(parallel, root, "library")
    store.index(parallel)
    assert pools == [(2, "spawn")]
    assert index_contents(parallel) == index_contents(db)
