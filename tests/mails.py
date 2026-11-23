from collections.abc import Sequence
from email import policy
from email.message import EmailMessage
from pathlib import Path

from sift.api.collections import TitleFrom
from sift.domain.text.documents import Convert, FileRead, Unread
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
