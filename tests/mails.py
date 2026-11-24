from collections.abc import Sequence
from email import policy
from email.message import EmailMessage
from pathlib import Path

from hypothesis import strategies as st

from sift.api.collections import TitleFrom
from sift.domain.text.documents import Conversion, Convert, FileRead, Unread
from sift.domain.text.notes import Document
from sift.ingest.documents import convert
from sift.ingest.readers import read_whole

SAMPLES = Path(__file__).parent / "samples"
WORD = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PARAGRAPH = "Kubelka-Munk theory predicts the colour of mixed paints from how each pigment absorbs and scatters light."

type Part = tuple[str | None, bytes, str] | tuple[str | None, EmailMessage]


def sample(extension: str) -> Path:
    return SAMPLES / f"sample{extension}"


def message(
    *,
    body: str | None = "Plain words.",
    html: str | None = None,
    attachments: Sequence[Part] = (),
    headers: dict[str, str] | None = None,
) -> EmailMessage:
    found = EmailMessage()
    for name, value in (headers or {}).items():
        found[name] = value
    if body is not None:
        found.set_content(body)
    if html is not None:
        if body is None:
            found.set_content(html, subtype="html")
        else:
            found.add_alternative(html, subtype="html")
    for part in attachments:
        match part:
            case (name, EmailMessage() as inner):
                found.add_attachment(inner, filename=name)
            case (name, bytes() as data, kind):
                main, sub = kind.split("/")
                found.add_attachment(data, maintype=main, subtype=sub, filename=name)
    return found


def nested_parts(depth: int) -> bytes:
    opened = (b'Content-Type: multipart/mixed; boundary="b%d"\r\n\r\n--b%d\r\n' % (n, n) for n in range(depth))
    closed = (b"--b%d--\r\n" % n for n in reversed(range(depth)))
    return b"".join((b"Subject: deep\r\nMIME-Version: 1.0\r\n", *opened, b"\r\nwords\r\n", *closed))


def mail_bytes(found: EmailMessage) -> bytes:
    return found.as_bytes(policy=policy.SMTP)


def read(raw: bytes, name: str, converting: Convert = convert) -> FileRead | Unread:
    return read_whole(raw, name, TitleFrom.HEADING, converting)


def read_file(raw: bytes, name: str, converting: Convert = convert) -> FileRead:
    found = read(raw, name, converting)
    assert isinstance(found, FileRead)
    return found


def only(raw: bytes, name: str, converting: Convert = convert) -> Document:
    found = read_file(raw, name, converting)
    assert found.problems == ()
    (document,) = found.documents
    return document


def failing(_raw: bytes, _name: str) -> Conversion:
    return Unread("Parsing error: HTML conversion failed: Internal panic: begin > end (2 > 0)")


HEADERS = (b"From: ", b"To: ", b"Cc: ", b"Subject: ", b"Date: ", b"Message-ID: ", b"Content-Type: ")
VALUES = (
    b"message/rfc822",
    b"message/delivery-status",
    b"multipart/mixed; boundary=b",
    b"multipart/mixed; boundary=c",
    b"text/plain; charset=",
    b"utf-7",
    b"+2D0-",
    b"x",
)
MARKS = (b"=?utf-8?q?", b"?=", b"<", b">", b"@", b",", b";", b":", b'"', b"(", b")", b"[", b"]", b"\\", b" ")
LINES = (
    b"\r\n",
    b"\r\n\r\n",
    b"\n",
    b"\r",
    b"--b\r\n",
    b"--b--",
    b"--c\r\n",
    b"words\r\n",
    b"\xff",
    b"\x00",
    b"Content-Transfer-Encoding: base64",
)
atoms = st.sampled_from((*HEADERS, *VALUES, *MARKS, *LINES, b"Content-Disposition: attachment; filename=a.md"))
