import json
from pathlib import Path
from typing import final, override

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.text.decoding import encodable
from sift.domain.text.documents import FileRead, Unread
from sift.ingest.mail import as_utf8, in_charset, mail, plain
from sift.ingest.readers import unique
from tests.mails import PARAGRAPH, atoms, failing, mail_bytes, message, only, read, read_file, sample

NO_CHARSET = b"Content-Type:multipart/;boundary=b\r--b\nContent-Type:charset=\x00"
OCTETS = "application/octet-stream"
MIXED = b"Subject: x\r\nMIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=b\r\n\r\n"
ATTACHED_TEXT = "Content-Disposition: attachment; filename=note.txt\r\n"


def _part(charset: str, disposition: str, text: bytes) -> bytes:
    head = f"--b\r\nContent-Type: text/plain; charset={charset}\r\n{disposition}"
    return head.encode() + b"Content-Transfer-Encoding: 8bit\r\n\r\n" + text + b"\r\n"


def test_a_mail_is_a_document_of_its_headers_and_its_plain_text() -> None:
    found = only(sample(".eml").read_bytes(), "sample.eml")
    assert found.title == "Pigment mixing"
    assert found.fields == {
        "from": "ada@example.com",
        "to": ["bo@example.org", "cy@example.org"],
        "cc": ["di@example.net"],
        "date": "2024-05-01T12:00:00+00:00",
        "message_id": "<pigments@example.com>",
    }
    head = (
        "# Pigment mixing\n\nFrom: Ada Example <ada@example.com>\nTo: Bo Reader <bo@example.org>, cy@example.org\n"
        "Cc: Di Copy <di@example.net>\nDate: 2024-05-01T12:00:00+00:00\n\nKubelka-Munk theory\n\n"
    )
    assert found.text.startswith(head)
    assert PARAGRAPH in found.text
    assert "<h1>" not in found.text


@pytest.mark.parametrize("empty", ["", " \n\n", "\r\n"])
def test_a_mail_whose_plain_part_is_empty_reads_its_html(empty: str) -> None:
    raw = mail_bytes(message(body=empty, html="<p>Real content here, xyzzy.</p>", headers={"Subject": "Offer"}))
    assert "Real content here, xyzzy." in only(raw, "a.eml").text


def test_a_mail_of_html_alone_reads_as_markdown() -> None:
    raw = mail_bytes(message(body=None, html="<h1>Offer</h1><p>Prices for <b>2027</b>.</p>", headers={"Subject": "O"}))
    text = only(raw, "a.eml").text
    assert "Prices for" in text
    assert "<p>" not in text


def test_headers_as_mail_writes_them() -> None:
    subject = "Grüße from Zürich, and a line so long that a mail client folds it, as mail clients fold long lines"
    headers = {
        "Subject": subject,
        "From": "Jürg Bühler <jb@example.ch>",
        "Date": "Tue, 14 Oct 2026 09:30:00 +0200",
        "In-Reply-To": "<earlier@example.ch>",
    }
    found = only(mail_bytes(message(headers=headers)), "a.eml")
    assert found.title == subject
    assert found.fields == {
        "from": "jb@example.ch",
        "date": "2026-10-14T07:30:00+00:00",
        "in_reply_to": "<earlier@example.ch>",
    }
    assert "From: Jürg Bühler <jb@example.ch>" in found.text
    bare = only(mail_bytes(message(body=None)), "Bare note.eml")
    assert (bare.title, bare.fields) == ("Bare note", {})
    assert only(b"Date: sometime soon\r\n\r\nPlain words.\r\n", "odd.eml").fields == {"date": "sometime soon"}


def test_a_subject_with_a_line_break_is_one_line() -> None:
    found = only(b"Subject: =?utf-8?q?Offer=0A_2026?=\r\n\r\nPlain words.\r\n", "offer.eml")
    assert (found.title, found.text) == ("Offer 2026", "# Offer 2026\n\nPlain words.")


def test_dates_sort_as_time_whatever_the_zone() -> None:
    dates = [
        only(mail_bytes(message(headers={"Date": date})), "a.eml").fields["date"]
        for date in (
            "Tue, 14 Oct 2026 09:30:00 +0200",
            "Tue, 14 Oct 2026 08:00:00 +0000",
            "Tue, 14 Oct 2026 08:00:00 -0000",
            "Fri, 31 Dec 9999 23:00:00 -0500",
        )
    ]
    assert dates == [
        "2026-10-14T07:30:00+00:00",
        "2026-10-14T08:00:00+00:00",
        "2026-10-14T08:00:00",
        "9999-12-31T23:00:00-05:00",
    ]


def test_raw_8_bit_addresses_and_text_of_half_characters_read_whole() -> None:
    raw = (
        b"From: J\xf6rg <j\xf6rg@example.com>\r\nTo: Ann <ann@example.com>, B\xe4r <b\xe4r@example.com>\r\n"
        b"Subject: Hello\r\nContent-Type: text/plain; charset=utf-7\r\n\r\nwombat +2D0- half an emoji\r\n"
    )
    found = only(raw, "old.eml")
    assert found.fields["from"] == "j\N{REPLACEMENT CHARACTER}rg@example.com"
    assert found.fields["to"] == ["ann@example.com", "b\N{REPLACEMENT CHARACTER}r@example.com"]
    assert "wombat \N{REPLACEMENT CHARACTER} half an emoji" in found.text
    assert found.text.encode()


@pytest.mark.parametrize(
    "charset", ["x-unknown", "undefined", "idna", "punycode", "unicode_escape", "raw-unicode-escape", "hex"]
)
@pytest.mark.parametrize("body", [b"Plain Gr\xc3\xbc\xc3\x9fe, C:\\new.", b"Plain words-only."])
def test_a_mail_in_a_charset_that_is_none_still_reads(charset: str, body: bytes) -> None:
    raw = b'Subject: x\r\nContent-Type: text/plain; charset="%s"\r\n\r\n%s\r\n' % (charset.encode(), body)
    assert body.decode() in only(raw, "x.eml").text


@pytest.mark.parametrize("charset", ["us-ascii", "utf-8"])
def test_text_that_is_not_of_the_charset_it_says_reads_as_ansi(charset: str) -> None:
    text = b"Gr\xfc\xdfe von M\xfcller."
    raw = MIXED + _part(charset, "", text) + _part(charset, ATTACHED_TEXT, text) + b"--b--\r\n"
    found = read_file(raw, "m.eml")
    assert found.problems == ()
    (letter, note) = found.documents
    assert letter.text.endswith("Grüße von Müller.")
    assert (note.key, note.text) == ("note.txt", "Grüße von Müller.")


def test_text_in_a_charset_of_its_own_keeps_it_where_a_byte_is_not_of_it() -> None:
    text = "見積書を送ります。".encode("shift_jis") + b"\x87\x40" + "以上です。".encode("shift_jis")
    raw = MIXED + _part("shift_jis", "", text) + _part("shift_jis", ATTACHED_TEXT, text) + b"--b--\r\n"
    for found in read_file(raw, "m.eml").documents:
        assert "見積書を送ります。" in found.text
        assert found.text.endswith("以上です。")


def test_a_text_attachment_reads_in_the_charset_its_mail_declares() -> None:
    def attached(name: str, charset: str, data: bytes) -> bytes:
        disposition = f"Content-Disposition: attachment; filename={name}\r\n"
        return _part(charset, disposition, data.removesuffix(b"\r\n"))

    raw = (
        MIXED
        + attached("latin.txt", "iso-8859-1", b"Gr\xfc\xdfe aus Z\xfcrich")
        + attached("ascii.txt", "us-ascii", "Grüße".encode())
        + attached("odd.txt", "x-unknown", b"Gr\xfc\xdfe")
        + b"--b--\r\n"
    )
    _, *notes = read_file(raw, "m.eml").documents
    assert [note.text for note in notes] == ["Grüße aus Zürich", "Grüße", "Grüße"]


def test_attachments_sift_reads_are_documents_of_their_own_named_once_each() -> None:
    raw = mail_bytes(
        message(
            headers={"Subject": "Contract"},
            attachments=[
                ("contract.pdf", sample(".pdf").read_bytes(), "application/pdf"),
                ("notes.md", b"# Notes\n\nSee the contract.\n", "text/markdown"),
                ("NOTES.md", b"# Notes, again\n", "text/markdown"),
                ("../../evil.md", b"# Evil\n", "text/markdown"),
                ("photo.png", b"\x89PNG", "image/png"),
                ("data.jsonl", b'{"text": "a"}\n', "application/json"),
                ("forwarded.eml", message(headers={"Subject": "inner"}, body="Inner.")),
            ],
        )
    )
    found = read_file(raw, "mail.eml")
    assert found.problems == ()
    keys = ["", "contract.pdf", "notes.md", "NOTES (2).md", "evil.md", "forwarded.eml"]
    assert [each.key for each in found.documents] == keys
    letter, pdf, *_ = found.documents
    names = ["contract.pdf", "notes.md", "NOTES.md", "evil.md", "photo.png", "data.jsonl", "forwarded.eml"]
    assert letter.fields["attachments"] == names
    assert f"Attachments: {', '.join(names)}" in letter.text
    assert PARAGRAPH in pdf.text
    assert pdf.digest != letter.digest


def test_names_taken_are_made_unique_with_a_few_looks_a_name() -> None:
    looked: list[int] = []

    @final
    class Counted(dict[str, int]):
        __hash__ = None

        @override
        def __contains__(self, key: object, /) -> bool:
            looked.append(1)
            return super().__contains__(key)

    taken = Counted()
    names = [unique(name, taken) for name in ("a (3).txt", *["A.txt"] * 2_000)]
    assert names[:5] == ["a (3).txt", "A.txt", "A (2).txt", "A (4).txt", "A (5).txt"]
    assert len({name.casefold() for name in names}) == len(names)
    assert len(looked) <= 3 * len(names)


def test_control_characters_in_an_attachment_s_name_are_dashes() -> None:
    disposition = b"Content-Disposition: attachment; filename*=utf-8''a%00b%0Ac%7F.md\r\n"
    raw = MIXED + b"--b\r\nContent-Type: text/markdown\r\n" + disposition + b"\r\n# A\r\n--b--\r\n"
    found = read_file(raw, "m.eml")
    assert [each.key for each in found.documents] == ["", "a-b-c-.md"]
    assert found.documents[0].fields["attachments"] == ["a-b-c-.md"]


def test_an_attachment_named_like_an_extension_or_not_named_is_passed_over() -> None:
    dotted = read_file(mail_bytes(message(attachments=[(".md", b"# Dot\n", "text/markdown")])), "a.eml")
    assert ([each.key for each in dotted.documents], dotted.documents[0].fields["attachments"]) == ([""], [".md"])
    unnamed = only(mail_bytes(message(attachments=[(None, b"# Notes\n", "text/markdown")])), "a.eml")
    assert unnamed.fields == {}


def test_a_mail_attached_to_a_mail_is_read_as_one_with_its_attachments() -> None:
    original = message(
        headers={"Subject": "Offer 2/2026", "From": "Anna <anna@acme.example>"},
        body="The original words.",
        attachments=[("contract.pdf", sample(".pdf").read_bytes(), "application/pdf")],
    )
    saved = b"Subject: Saved\r\n\r\nSaved words.\r\n"
    raw = mail_bytes(
        message(
            headers={"Subject": "Fwd: Offer"},
            attachments=[(None, original), ("saved.eml", saved, "application/octet-stream"), (None, message())],
        )
    )
    found = read_file(raw, "mail.eml")
    keys = ["", "Offer 2-2026.eml", "Offer 2-2026.eml/contract.pdf", "saved.eml", "message.eml"]
    assert ([each.key for each in found.documents], found.problems) == (keys, ())
    letter, forwarded, pdf, kept, _ = found.documents
    assert letter.fields["attachments"] == ["Offer 2-2026.eml", "saved.eml", "message.eml"]
    assert (forwarded.title, forwarded.fields["from"]) == ("Offer 2/2026", "anna@acme.example")
    assert forwarded.text.endswith("The original words.")
    assert PARAGRAPH in pdf.text
    assert kept.text == "# Saved\n\nSaved words."


def test_mail_in_mail_is_read_three_levels_deep() -> None:
    inner = message(headers={"Subject": "0"})
    for level in range(1, 6):
        inner = message(headers={"Subject": str(level)}, attachments=[(None, inner)])
    found = read_file(mail_bytes(message(headers={"Subject": "top"}, attachments=[(None, inner)])), "a.eml")
    keys = ["", "5.eml", "5.eml/4.eml", "5.eml/4.eml/3.eml"]
    assert ([each.key for each in found.documents], found.problems) == (keys, ())
    assert found.documents[-1].fields["attachments"] == ["2.eml"]


def test_a_damaged_attachment_is_a_problem_and_the_mail_still_reads() -> None:
    raw = mail_bytes(
        message(headers={"Subject": "Scan"}, attachments=[("scan.pdf", b"%PDF-1.4 broken", "application/pdf")])
    )
    found = read_file(raw, "a.eml")
    assert [each.key for each in found.documents] == [""]
    (problem,) = found.problems
    assert problem.startswith("scan.pdf: Parsing error: ")


def test_a_damaged_mail_attached_is_a_problem_and_the_mail_still_reads() -> None:
    raw = mail_bytes(message(headers={"Subject": "Fwd"}, attachments=[("saved.eml", NO_CHARSET, OCTETS)]))
    found = read_file(raw, "a.eml")
    assert [each.key for each in found.documents] == [""]
    assert found.problems == ("saved.eml: embedded null character",)


def test_a_mail_of_attachments_alone_is_its_subject_and_their_names() -> None:
    raw = mail_bytes(message(body=None, headers={"Subject": "Scan"}, attachments=[("a.md", b"# A\n", "text/markdown")]))
    assert read_file(raw, "a.eml").documents[0].text == "# Scan\n\nAttachments: a.md"


def test_a_mail_whose_html_has_no_text_says_so() -> None:
    raw = mail_bytes(message(body=None, html="<p> </p><div></div>", headers={"Subject": "Offer"}))
    found = read_file(raw, "a.eml")
    assert (found.documents[0].text, found.problems) == ("# Offer", ("its HTML: no text in it",))


def test_html_the_converter_fails_on_is_read_as_its_words() -> None:
    html = (
        "<html><head><title>T</title><style>p {color: red}</style></head><body><h1>Offer</h1>"
        "<p>Prices for <b>2027</b> &amp; later.</p><ul><li>One</li><li>Two</li></ul><script>run()</script>"
        "</body></html>"
    )
    found = read_file(mail_bytes(message(body=None, html=html, headers={"Subject": "Offer"})), "a.eml", failing)
    assert found.documents[0].text.endswith("Offer\n\nPrices for 2027 & later.\n\nOne\n\nTwo")
    said = (
        "its HTML: read as plain text, as converting it failed (Parsing error: HTML conversion failed: Internal "
        "panic: begin > end (2 > 0))"
    )
    assert found.problems == (said,)


def test_a_mail_whose_charset_names_no_charset_is_unread_with_what_python_said() -> None:
    assert mail(NO_CHARSET, "a.eml", failing) == Unread("embedded null character")


@pytest.mark.parametrize(
    "raw",
    [
        b"From::;R",
        b"From:, :",
        b"From:s@[",
        b"From: A]a <a@b.c@d.e>, c@d.e\r\nMessage-ID: <p@[\nexample.com>\r\n\r\nx\r\n",
    ],
)
def test_headers_python_s_parser_failed_on_leave_a_mail_read_or_unread_by_python_s_version(raw: bytes) -> None:
    found = mail(raw, "a.eml", failing)
    assert isinstance(found, Unread) or found.document.title == "a"


@given(st.lists(atoms, max_size=40).map(b"".join))
def test_any_mail_is_read_or_unread_and_what_is_read_can_be_stored(raw: bytes) -> None:
    found = read(raw, "a.eml", failing)
    if isinstance(found, FileRead):
        for each in found.documents:
            assert encodable(each.text)
            assert encodable(each.title)
            assert encodable(json.dumps(each.fields, ensure_ascii=False))


def test_charsets_and_html_words_on_their_own() -> None:
    assert (in_charset(b"\xfc", "latin-1"), in_charset(b"x", "x-unknown"), in_charset(b"\xfc", "ascii")) == (
        "ü",
        None,
        None,
    )
    assert in_charset(b"\x87\x40", "shift_jis") == "\N{REPLACEMENT CHARACTER}@"
    assert (as_utf8(b"\xfc", None), as_utf8("ü".encode(), "latin-1"), as_utf8(b"\xfc", "latin-1")) == (
        b"\xfc",
        "ü".encode(),
        "ü".encode(),
    )
    assert plain("<p>a <i>b</i></p><script>c</script><br>d") == "a b\n\nd"


def test_a_mail_saved_as_a_file_is_read_by_its_name(tmp_path: Path) -> None:
    path = tmp_path / "Offer.EML"
    path.write_bytes(mail_bytes(message(headers={"Subject": "Offer"})))
    assert only(path.read_bytes(), path.name).title == "Offer"
