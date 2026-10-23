"""Reading files by extension: one reader per kind of file, and PDF, office and other documents converted to Markdown.

Every extension sift reads has a sample in tests/samples, all with the same content (see make_samples.py), so one
contract covers every reader."""

import json
import re
import shutil
import zipfile
from pathlib import Path

import pytest

from sift import indexing, readers, store
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
    (doc,), problems = readers.load(sample(ext))
    assert problems == []
    return doc


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


# --- the contract every reader keeps -----------------------------------------


def test_every_extension_sift_reads_has_a_sample():
    assert {p.suffix for p in SAMPLES.glob("sample.*") if p.suffix != ".py"} == set(readers.READERS)


@pytest.mark.parametrize("ext", sorted(readers.READERS))
def test_every_reader_keeps_the_contract(ext):
    docs, problems = readers.load(sample(ext))
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
    assert readers.load(sample(ext)) == (docs, problems)  # the same file reads the same every time


def test_extensions_match_in_any_case(tmp_path):
    shutil.copy(sample(".pdf"), tmp_path / "SCAN.PDF")
    shutil.copy(sample(".md"), tmp_path / "Notes.MD")
    assert readers.load(tmp_path / "SCAN.PDF") == readers.load(sample(".pdf"))
    assert readers.load(tmp_path / "Notes.MD")[0][0].text == read(".md").text
    assert sorted(Path(p).name for p, _, _ in indexing.walk(str(tmp_path))) == ["Notes.MD", "SCAN.PDF"]


def test_a_new_kind_of_file_needs_only_a_reader(tmp_path, home, monkeypatch):
    def read_csv(raw, name, fields):
        text = raw.decode()
        return [readers.Doc("", "digest", name, {}, text, list(readers.chunk(text)))], []

    monkeypatch.setitem(readers.READERS, ".csv", read_csv)
    write(tmp_path / "c" / "pigments.csv", "pigment,index\nUltramarine,PB29\n")
    db = store.connect(home)
    store.add_collection(db, tmp_path / "c", "c")
    indexing.index(db)
    assert Index(db).get("c/pigments.csv") == "pigment,index\nUltramarine,PB29\n"


# --- documents: PDF, office files, e-books and web pages --------------------------


@pytest.mark.parametrize("ext", sorted(set(readers.DOCUMENTS) - set(PAGES)))
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
    docs, (problem,) = readers.load(tmp_path / name)
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
    docs, problems = readers.load(path)
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
    docs, (problem,) = readers.load(path)
    assert docs == []
    assert "exceeds limit" in problem


def test_a_pdf_without_text_is_passed_on_as_a_scan():
    # Readers run in worker processes: the process that writes the index reads the scan with OCR (see test_ocr).
    (doc,), problems = readers.load(SAMPLES / "blank.pdf")
    assert (doc.scan, doc.text, doc.chunks, problems) == ((SAMPLES / "blank.pdf").read_bytes(), "", [], [])


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
    indexing.index(db)
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
    indexing.index(db)
    assert "library: broken.pdf: skipped (" in capsys.readouterr().err
    indexing.index(db)
    assert "broken.pdf" not in capsys.readouterr().err  # unchanged since, so not read again
    shutil.copy(sample(".pdf"), root / "broken.pdf")
    indexing.index(db)
    assert "1 added" in capsys.readouterr().err
    assert Index(db).get("library/broken.pdf") == read(".pdf").text


def test_documents_are_converted_in_worker_processes(library, tmp_path, monkeypatch, pools):
    db, root = library
    monkeypatch.setattr(indexing, "_cpus", lambda: 3)  # two workers, one CPU for writing
    monkeypatch.setattr(indexing, "FILES_PER_WORKER", 1)
    parallel = store.connect(tmp_path / "parallel")
    store.add_collection(parallel, root, "library")
    indexing.index(parallel)
    assert pools == [(2, "spawn")]
    assert index_contents(parallel) == index_contents(db)


def test_documents_an_older_sift_skipped_are_read_again_once(tmp_path, home, monkeypatch):
    # The server of sift 1.3.0 and older skipped every document its reindex tool was given: their files, and mails
    # with one attached, are read again by the next index.
    root, word = tmp_path / "docs", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    root.mkdir()
    shutil.copy(sample(".docx"), root / "offer.docx")
    email_file(root, Subject="Offer", attachments=[("offer.docx", sample(".docx").read_bytes(), word)])
    db = store.connect(home)
    store.add_collection(db, root, "docs")
    with monkeypatch.context() as patched:
        patched.setattr(readers, "_xberg", lambda raw, name: 1 / 0)
        indexing.index(db)  # as that server did: no documents of them
    assert [p for (p,) in db.execute("SELECT path FROM docs")] == ["mail.eml"]
    db.execute("DELETE FROM meta WHERE key = 'documents_read'")  # as an index of sift 1.3.0 has it
    db.commit()
    db.close()
    db = store.connect(home)
    assert db.execute("SELECT count(*) FROM sources WHERE mtime IS NULL").fetchone() == (2,)  # the file and the mail
    indexing.index(db)
    assert sorted(p for (p,) in db.execute("SELECT path FROM docs")) == ["mail.eml", "mail/offer.docx", "offer.docx"]
    db.close()
    assert store.connect(home).execute("SELECT count(*) FROM sources WHERE mtime IS NULL").fetchone() == (0,)


# --- saved emails ------------------------------------------------------------


def message(*, body="Plain words.", html=None, attachments=(), **headers):
    """An email with these headers (From, To, Subject, ...), a plain body and, if given, its HTML alternative, and
    attachments: (file name, bytes, type), or (file name or None, a message) for a mail attached as message/rfc822."""
    from email.message import EmailMessage

    msg = EmailMessage()
    for key, value in headers.items():
        msg[key.replace("_", "-")] = value
    if body is not None:
        msg.set_content(body)
    if html is not None:
        if body is None:
            msg.set_content(html, subtype="html")
        else:
            msg.add_alternative(html, subtype="html")
    for filename, data, *kind in attachments:
        if kind:
            main, sub = kind[0].split("/")
            msg.add_attachment(data, maintype=main, subtype=sub, filename=filename)
        else:
            msg.add_attachment(data, filename=filename)
    return msg


def email_file(tmp_path, *, name="mail.eml", **parts):
    """An .eml file of the message these parts make (see `message`)."""
    from email import policy

    path = tmp_path / name
    path.write_bytes(message(**parts).as_bytes(policy=policy.SMTP))
    return path


def test_an_email_is_a_document_of_its_headers_and_text():
    doc = read(".eml")
    assert doc.title == "Pigment mixing"
    assert doc.meta == {
        "from": "ada@example.com",
        "to": ["bo@example.org", "cy@example.org"],
        "cc": ["di@example.net"],
        "date": "2024-05-01T12:00:00+00:00",
        "message_id": "<pigments@example.com>",
    }
    assert doc.text.startswith(
        "# Pigment mixing\n\nFrom: Ada Example <ada@example.com>\nTo: Bo Reader <bo@example.org>, cy@example.org\n"
        "Cc: Di Copy <di@example.net>\nDate: 2024-05-01T12:00:00+00:00\n\nKubelka-Munk theory\n\n"
    )
    assert "<h1>" not in doc.text  # its plain text, not the HTML beside it


def test_an_email_of_html_alone_reads_as_markdown(tmp_path):
    path = email_file(tmp_path, body=None, html="<h1>Offer</h1><p>Prices for <b>2027</b>.</p>", Subject="Offer")
    (doc,), _ = readers.load(path)
    assert "Prices for" in doc.text
    assert "<p>" not in doc.text


def test_email_headers_as_mail_writes_them(tmp_path):
    # Encoded words, folded lines, a date in another zone; and a mail without subject, date or body.
    path = email_file(
        tmp_path,
        Subject="Grüße from Zürich, and a line so long that a mail client folds it, as mail clients fold long lines",
        From="Jürg Bühler <jb@example.ch>",
        Date="Tue, 14 Oct 2026 09:30:00 +0200",
        In_Reply_To="<earlier@example.ch>",
    )
    (doc,), _ = readers.load(path)
    assert (
        doc.title
        == "Grüße from Zürich, and a line so long that a mail client folds it, as mail clients fold long lines"
    )
    assert doc.meta == {
        "from": "jb@example.ch",
        "date": "2026-10-14T07:30:00+00:00",
        "in_reply_to": "<earlier@example.ch>",
    }
    assert "From: Jürg Bühler <jb@example.ch>" in doc.text
    (bare,), problems = readers.load(email_file(tmp_path, name="Bare note.eml", body=None))
    assert (bare.title, bare.meta, problems) == ("Bare note", {}, [])
    (tmp_path / "odd.eml").write_bytes(b"Date: sometime soon\r\n\r\nPlain words.\r\n")  # EmailMessage drops such dates
    (odd,), _ = readers.load(tmp_path / "odd.eml")
    assert odd.meta == {"date": "sometime soon"}


def test_mail_dates_sort_as_time_whatever_the_zone(tmp_path):
    # With the sender's offset, 09:30 in Zurich sorted after 08:00 in London, though it was sent first.
    zurich = email_file(tmp_path, name="a.eml", Date="Tue, 14 Oct 2026 09:30:00 +0200")
    london = email_file(tmp_path, name="b.eml", Date="Tue, 14 Oct 2026 08:00:00 +0000")
    unknown = email_file(tmp_path, name="c.eml", Date="Tue, 14 Oct 2026 08:00:00 -0000")  # a zone unknown, RFC 5322
    dates = [readers.load(path)[0][0].meta["date"] for path in (zurich, london, unknown)]
    assert dates == ["2026-10-14T07:30:00+00:00", "2026-10-14T08:00:00+00:00", "2026-10-14T08:00:00"]


def test_an_email_in_an_unknown_charset_still_reads(tmp_path):
    path = tmp_path / "x.eml"
    path.write_bytes(
        b'Subject: x\r\nContent-Type: text/plain; charset="x-unknown"\r\n\r\nPlain Gr\xc3\xbc\xc3\x9fe.\r\n'
    )
    (doc,), _ = readers.load(path)
    assert "Plain Grüße." in doc.text


def test_a_text_attachment_reads_in_the_charset_its_mail_declares(tmp_path):
    # sift reads text files as UTF-8; a mail can say that an attachment is Latin-1, whose umlauts came out as "�".
    def attachment(name, charset, data):
        return (
            f"--b\r\nContent-Type: text/plain; charset={charset}\r\n"
            f"Content-Disposition: attachment; filename={name}\r\nContent-Transfer-Encoding: 8bit\r\n\r\n"
        ).encode() + data

    head = b"Subject: x\r\nMIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=b\r\n\r\n"
    (tmp_path / "m.eml").write_bytes(
        head
        + attachment("latin.txt", "iso-8859-1", b"Gr\xfc\xdfe aus Z\xfcrich\r\n")
        + attachment("ascii.txt", "us-ascii", "Grüße\r\n".encode())  # UTF-8 already, whatever the mail calls it
        + attachment("odd.txt", "x-unknown", b"Gr\xfc\xdfe\r\n")  # a charset Python doesn't know: as any text file
        + b"--b--\r\n"
    )
    (_, *attached), problems = readers.load(tmp_path / "m.eml")
    assert [doc.text for doc in attached] == ["Grüße aus Zürich", "Grüße", "Gr\ufffd\ufffde"]
    assert problems == []


def test_attachments_sift_reads_are_documents_of_their_own(tmp_path):
    path = email_file(
        tmp_path,
        Subject="Contract",
        attachments=[
            ("contract.pdf", sample(".pdf").read_bytes(), "application/pdf"),
            ("notes.md", b"# Notes\n\nSee the contract.\n", "text/markdown"),
            ("notes.md", b"# Notes, again\n", "text/markdown"),
            ("../../evil.md", b"# Evil\n", "text/markdown"),
            ("photo.png", b"\x89PNG", "image/png"),
            ("forwarded.eml", message(Subject="inner", body="Inner.")),
        ],
    )
    docs, problems = readers.load(path)
    assert problems == []
    assert [doc.key for doc in docs] == ["", "contract.pdf", "notes.md", "notes (2).md", "evil.md", "forwarded.eml"]
    mail, pdf = docs[:2]
    assert mail.meta["attachments"] == ["contract.pdf", "notes.md", "notes.md", "evil.md", "photo.png", "forwarded.eml"]
    assert "Attachments: contract.pdf, notes.md, notes.md, evil.md, photo.png, forwarded.eml" in mail.text
    assert PARAGRAPH in pdf.text
    assert pdf.digest != mail.digest  # unchanged attachments keep their vectors when the mail changes


def test_an_attachment_named_like_an_extension_has_none(tmp_path):
    # As a file named `.md` has none; looking up its reader by an empty extension threw the mail away.
    path = email_file(tmp_path, Subject="Odd", attachments=[(".md", b"# Dot\n", "text/markdown")])
    (mail,), problems = readers.load(path)
    assert (mail.meta["attachments"], problems) == ([".md"], [])


def test_a_mail_attached_to_a_mail_is_read_as_one(tmp_path):
    # As mail clients forward a mail "as attachment": its headers, text and attachments are documents too, named by
    # its subject if it has no file name; and a mail saved as .eml and attached as a file reads the same.
    original = message(
        Subject="Offer 2/2026",
        From="Anna <anna@acme.example>",
        body="The original words.",
        attachments=[("contract.pdf", sample(".pdf").read_bytes(), "application/pdf")],
    )
    saved = b"Subject: Saved\r\n\r\nSaved words.\r\n"
    path = email_file(
        tmp_path,
        Subject="Fwd: Offer",
        attachments=[(None, original), ("saved.eml", saved, "application/octet-stream"), (None, message(body="x"))],
    )
    docs, problems = readers.load(path)
    keys = ["", "Offer 2-2026.eml", "Offer 2-2026/contract.pdf", "saved.eml", "message.eml"]
    assert ([doc.key for doc in docs], problems) == (keys, [])
    mail, forwarded, pdf, saved_doc, _ = docs
    assert mail.meta["attachments"] == ["Offer 2-2026.eml", "saved.eml", "message.eml"]
    assert (forwarded.title, forwarded.meta["from"]) == ("Offer 2/2026", "anna@acme.example")
    assert forwarded.text.endswith("The original words.")
    assert PARAGRAPH in pdf.text
    assert saved_doc.text == "# Saved\n\nSaved words."


def test_mail_in_mail_is_read_three_levels_deep(tmp_path):
    # A mail that nests mails without end must not take sift down with it; the deepest it reads lists the next.
    inner = message(Subject="0")
    for level in range(1, 6):
        inner = message(Subject=str(level), attachments=[(None, inner)])
    docs, problems = readers.load(email_file(tmp_path, Subject="top", attachments=[(None, inner)]))
    assert ([doc.key for doc in docs], problems) == (["", "5.eml", "5/4.eml", "5/4/3.eml"], [])
    assert docs[-1].meta["attachments"] == ["2.eml"]


def test_a_damaged_mail_attached_is_reported_and_the_mail_still_reads(tmp_path, monkeypatch):
    path = email_file(tmp_path, Subject="Fwd", attachments=[(None, message(Subject="inner"))])
    real = readers._mail_headers
    monkeypatch.setattr(readers, "_mail_headers", lambda msg: real(msg) if msg["subject"] == "Fwd" else 1 / 0)
    docs, problems = readers.load(path)
    assert ([doc.key for doc in docs], problems) == ([""], ["inner.eml: skipped (division by zero)"])


def test_an_attachment_without_a_name_is_passed_over(tmp_path):
    # Its document would be named by its file name; but for a mail, it has nothing else to be named by.
    path = email_file(tmp_path, Subject="x", attachments=[(None, b"# Notes\n", "text/markdown")])
    (mail,), problems = readers.load(path)
    assert (mail.meta, problems) == ({}, [])


def test_an_email_of_attachments_alone(tmp_path):
    path = email_file(tmp_path, body=None, Subject="Scan", attachments=[("a.md", b"# A\n", "text/markdown")])
    (mail, _), _ = readers.load(path)
    assert mail.text == "# Scan\n\nAttachments: a.md"


def test_a_mail_whose_html_has_no_text_says_so(tmp_path):
    # The mail stays, with its headers; that its text is missing is reported, not passed over.
    (mail,), problems = readers.load(email_file(tmp_path, body=None, html="<p> </p><div></div>", Subject="Offer"))
    assert (mail.text, problems) == ("# Offer", ["its HTML: no text in it"])


def test_html_the_converter_fails_on_is_read_as_plain_text(tmp_path, monkeypatch):
    # Xberg's converter panicked on some marketing mails ("begin > end (2 > 0)"), and their text was lost.
    def panic(raw, name):
        raise RuntimeError("Parsing error: HTML conversion failed: Internal panic: begin > end (2 > 0)")

    monkeypatch.setattr(readers, "_xberg", panic)
    html = (
        "<html><head><title>T</title><style>p {color: red}</style></head><body><h1>Offer</h1>"
        "<p>Prices for <b>2027</b> &amp; later.</p><ul><li>One</li><li>Two</li></ul><script>run()</script>"
        "</body></html>"
    )
    (mail,), problems = readers.load(email_file(tmp_path, body=None, html=html, Subject="Offer"))
    assert mail.text.endswith("Offer\n\nPrices for 2027 & later.\n\nOne\n\nTwo")
    assert problems == [
        "its HTML: read as plain text, as converting it failed (Parsing error: HTML conversion failed: Internal "
        "panic: begin > end (2 > 0))"
    ]


def test_a_mail_the_parser_fails_on_is_skipped(tmp_path, monkeypatch):
    # Which input breaks Python's email parser changes with Python's version; that one does must not stop indexing.
    import email

    def failing(raw, policy):
        raise ValueError("header line 3: no colon")

    monkeypatch.setattr(email, "message_from_bytes", failing)
    assert readers.load(email_file(tmp_path)) == ([], ["skipped (header line 3: no colon)"])


def test_a_damaged_attachment_is_reported_and_the_mail_still_reads(tmp_path):
    path = email_file(tmp_path, Subject="Scan", attachments=[("scan.pdf", b"%PDF-1.4 broken", "application/pdf")])
    docs, problems = readers.load(path)
    assert [doc.key for doc in docs] == [""]
    assert [problem[:17] for problem in problems] == ["scan.pdf: skipped"]


def test_a_folder_of_mail_is_searched_and_filtered(tmp_path, home):
    (tmp_path / "mail").mkdir()
    email_file(
        tmp_path / "mail",
        name="offer.eml",
        Subject="Offer",
        From="Ada Example <ada@example.com>",
        attachments=[("contract.pdf", sample(".pdf").read_bytes(), "application/pdf")],
    )
    db = store.connect(home)
    store.add_collection(db, tmp_path / "mail", "mail")
    indexing.index(db)
    index = Index(db)
    assert [h["id"] for h in index.search("Kubelka-Munk")] == ["mail/offer/contract.pdf"]
    assert [h["id"] for h in index.search("offer", filters={"from": "ada@example.com"})] == ["mail/offer.eml"]
