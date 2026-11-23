import codecs
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email import policy
from email.errors import MessageError
from email.headerregistry import Address, AddressHeader, DateHeader
from email.message import EmailMessage, Message, MIMEPart
from email.parser import BytesParser
from html.parser import HTMLParser
from typing import final, override

from sift.api.fields import Fields, FieldValue
from sift.domain.text.chunks import chunks
from sift.domain.text.decoding import decoded
from sift.domain.text.documents import Convert, Unread
from sift.domain.text.json_values import LONE_SURROGATE, REPLACEMENT, composed_fields
from sift.domain.text.kinds import CONTROL, FileKind, kind_of
from sift.domain.text.markdown import one_line
from sift.domain.text.notes import Document, digest, file_stem, storable
from sift.ingest.nesting import nesting

ADDRESSES = ("from", "to", "cc")
IDS = ("message-id", "in-reply-to")
NO_CHARSETS = frozenset({"idna", "punycode", "undefined", "unicode-escape", "raw-unicode-escape"})
SAYS_NOTHING = frozenset({"ascii", "utf-8"})
UNREADABLE = (MessageError, ValueError, LookupError, IndexError, AttributeError, NameError)
MOST_NESTED = 50
NESTED_TOO_DEEPLY = f"its parts nest more than {MOST_NESTED} deep"
ATTACHED = frozenset({FileKind.NOTE, FileKind.DOCUMENT, FileKind.MAIL})
ATTACHED_MAIL = "message/rfc822"
FOLDERS = re.compile(r"[\\/]")
HTML_NAME = "mail.html"
NOT_NAMES = frozenset({"", ".", ".."})
NAME_DASH = "-"
UNNAMED_MAIL = "message"
HIDDEN = frozenset({"head", "script", "style", "template"})
BLOCKS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)


@dataclass(frozen=True, slots=True)
class Attachment:
    name: str
    raw: bytes
    mail: bool


@dataclass(frozen=True, slots=True)
class Mail:
    document: Document
    attachments: tuple[Attachment, ...]
    problems: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Head:
    fields: Fields
    shown: tuple[str, ...]


@final
class _Words(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self.words: list[str] = []
        self.hidden: int = 0

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.hidden += tag in HIDDEN
        if tag in BLOCKS:
            self.end_paragraph()

    @override
    def handle_endtag(self, tag: str) -> None:
        if tag in HIDDEN and self.hidden:
            self.hidden -= 1
        if tag in BLOCKS:
            self.end_paragraph()

    @override
    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.words.append(data)

    def end_paragraph(self) -> None:
        if paragraph := " ".join("".join(self.words).split()):
            self.paragraphs.append(paragraph)
        self.words = []


def plain(html: str) -> str:
    words = _Words()
    words.feed(html)
    words.close()
    words.end_paragraph()
    return "\n\n".join(words.paragraphs)


def in_charset(raw: bytes, charset: str) -> str | None:
    try:
        if codecs.lookup(charset).name in NO_CHARSETS:
            return None
        return raw.decode(charset)
    except LookupError:
        return None
    except UnicodeError:
        return None if codecs.lookup(charset).name in SAYS_NOTHING else raw.decode(charset, "replace")


def _payload(part: Message) -> bytes:
    raw: object = part.get_payload(decode=True)
    return raw if isinstance(raw, bytes) else b""


def _text_of(part: Message) -> str:
    raw = _payload(part)
    text = in_charset(raw, part.get_content_charset("ascii"))
    return decoded(raw) if text is None else text


def _whole(text: str) -> str:
    return LONE_SURROGATE.sub(REPLACEMENT, text)


def _in_utc(when: datetime) -> datetime:
    try:
        return when.astimezone(UTC)
    except OverflowError:
        return when


def _date(header: object) -> str:
    when = header.datetime if isinstance(header, DateHeader) else None
    if when is None:
        return str(header)
    return (_in_utc(when) if when.tzinfo else when).isoformat()


def _head(message: EmailMessage) -> _Head:
    fields: dict[str, FieldValue] = {}
    shown: list[str] = []
    for name in ADDRESSES:
        header: object = message.get(name)
        found = [_whole(each.addr_spec) for each in _addresses(header) if each.addr_spec]
        if found:
            fields[name] = found[0] if name == "from" else found
            shown.append(_whole(f"{name.capitalize()}: {header}"))
    date: object = message.get("date")
    if date is not None:
        when = _whole(_date(date))
        fields["date"] = when
        shown.append(f"Date: {when}")
    for name in IDS:
        if value := _whole(str(message.get(name) or "").strip()):
            fields[name.replace("-", "_")] = value
    return _Head(composed_fields(fields), tuple(shown))


def _addresses(header: object) -> tuple[Address, ...]:
    return header.addresses if isinstance(header, AddressHeader) else ()


def _text(message: EmailMessage, convert: Convert) -> tuple[str, tuple[str, ...]]:
    plain_part = message.get_body(preferencelist=("plain",))
    if plain_part is not None and (text := _text_of(plain_part).replace("\r\n", "\n").strip()):
        return text, ()
    html_part = message.get_body(preferencelist=("html",))
    if html_part is None:
        return "", ()
    content = _whole(_text_of(html_part))
    converted = convert(content.encode(), HTML_NAME)
    if isinstance(converted, Unread):
        return plain(content), (f"its HTML: read as plain text, as converting it failed ({converted.problem})",)
    found = converted.markdown.strip()
    return found, () if found else ("its HTML: no text in it",)


def attached_mail(part: MIMEPart) -> MIMEPart | None:
    return next(part.iter_parts(), None) if part.get_content_type() == ATTACHED_MAIL else None


def attachment_name(part: MIMEPart) -> str | None:
    name = FOLDERS.split(part.get_filename() or "")[-1].strip()
    if name not in NOT_NAMES:
        return CONTROL.sub(NAME_DASH, name)
    inner = attached_mail(part)
    if inner is None:
        return None
    subject = CONTROL.sub(NAME_DASH, FOLDERS.sub(NAME_DASH, str(inner.get("subject") or ""))).strip(" .")
    return f"{subject or UNNAMED_MAIL}.eml"


def _utf8(raw: bytes) -> bool:
    try:
        raw.decode()
    except UnicodeDecodeError:
        return False
    return True


def as_utf8(raw: bytes, charset: str | None) -> bytes:
    if not charset or _utf8(raw):
        return raw
    text = in_charset(raw, charset)
    return raw if text is None else _whole(text).encode()


def _attachment(part: MIMEPart) -> Attachment | None:
    name = attachment_name(part)
    if name is None:
        return None
    inner = attached_mail(part)
    if inner is not None:
        return Attachment(name, inner.as_bytes(), mail=True)
    kind = kind_of(name)
    if kind not in ATTACHED:
        return None
    raw = _payload(part)
    if kind is FileKind.NOTE:
        return Attachment(name, as_utf8(raw, part.get_content_charset()), mail=False)
    return Attachment(name, raw, mail=kind is FileKind.MAIL)


def _attachments(parts: Sequence[MIMEPart]) -> Iterator[Attachment]:
    for part in parts:
        if (found := _attachment(part)) is not None:
            yield found


def _read(raw: bytes, name: str, convert: Convert) -> Mail:
    message = BytesParser(EmailMessage, policy=policy.default).parsebytes(raw)
    head = _head(message)
    parts = list(message.iter_attachments())
    names = [found for part in parts if (found := attachment_name(part))]
    fields = dict(head.fields)
    shown = list(head.shown)
    if names:
        fields["attachments"] = names
        shown.append("Attachments: " + ", ".join(names))
    title = one_line(_whole(str(message.get("subject") or ""))) or file_stem(name)
    body, problems = _text(message, convert)
    text = storable(_whole("\n\n".join(part for part in (f"# {title}", "\n".join(shown), body) if part)))
    document = Document("", digest(raw), storable(title), composed_fields(fields), text, chunks(text), ())
    return Mail(document, tuple(_attachments(parts)), problems)


def mail(raw: bytes, name: str, convert: Convert) -> Mail | Unread:
    try:
        if nesting(raw, MOST_NESTED) > MOST_NESTED:
            return Unread(NESTED_TOO_DEEPLY)
        return _read(raw, name, convert)
    except UNREADABLE as error:
        return Unread(str(error))
