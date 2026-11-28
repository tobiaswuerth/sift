import posixpath
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from sift.api.collections import TitleFrom
from sift.domain.text.documents import (
    Again,
    Convert,
    FileRead,
    Outcome,
    Refused,
    Scan,
    Unread,
    document,
    text_held,
    too_large,
)
from sift.domain.text.kinds import FileKind, kind_of
from sift.domain.text.links import folded
from sift.domain.text.notes import Document, note
from sift.ingest.documents import convert
from sift.ingest.files import Identity, at_most
from sift.ingest.mail import Attachment, mail

MAIL_DEPTH = 3
MOST_TEXT = 100_000_000

type ReadWhole = Callable[[Path, int, Identity | None], bytes]


@dataclass(frozen=True, slots=True)
class ToRead:
    path: Path
    size: int
    most: int
    scanned: Identity | None
    title_from: TitleFrom


def unique(name: str, taken: dict[str, int]) -> str:
    stem, extension = posixpath.splitext(name)
    named = folded(name)
    key, number = name, taken.get(named, 1)
    while folded(key) in taken:
        number += 1
        key = f"{stem} ({number}){extension}"
    taken[named] = number
    taken.setdefault(folded(key), 1)
    return key


def within(key: str, inner: str) -> str:
    return f"{key}/{inner}" if inner else key


@dataclass(frozen=True, slots=True)
class _Asked:
    title_from: TitleFrom
    convert: Convert
    most_text: int


def _attachment(attached: Attachment, asked: _Asked, depth: int) -> FileRead | Unread:
    if not attached.mail:
        return read_whole(attached.raw, attached.name, asked.title_from, asked.convert, asked.most_text)
    return _mail(attached.raw, attached.name, asked, depth - 1)


def _mail(raw: bytes, name: str, asked: _Asked, depth: int) -> FileRead | Unread:
    found = mail(raw, name, asked.convert)
    if isinstance(found, Unread):
        return found
    documents: list[Document] = [found.document]
    problems = list(found.problems)
    scans: list[Scan] = []
    taken: dict[str, int] = {}
    held, left_out = len(found.document.text), list[str]()
    for attached in found.attachments:
        if held > asked.most_text:
            left_out.append(attached.name)
            continue
        if attached.mail and not depth:
            problems.append(f"{attached.name}: a mail nested more than {MAIL_DEPTH} deep, left out")
            continue
        read = _attachment(attached, replace(asked, most_text=asked.most_text - held), depth)
        held += text_held(read)
        key = unique(attached.name, taken)
        match read:
            case Unread(problem=problem):
                problems.append(f"{attached.name}: {problem}")
            case _:
                documents += [replace(each, key=within(key, each.key)) for each in read.documents]
                problems += [f"{attached.name}: {problem}" for problem in read.problems]
                scans += [_keyed(each, within(key, each.document.key)) for each in read.scans]
    if left_out:
        said = f"left out, as the mail's text came to more than {asked.most_text:,} characters"
        problems.append(f"{', '.join(left_out)}: {said}")
    return FileRead(tuple(documents), tuple(problems), tuple(scans))


def _keyed(scan: Scan, key: str) -> Scan:
    return replace(scan, document=replace(scan.document, key=key))


def read_whole(
    raw: bytes, name: str, title_from: TitleFrom, convert: Convert, most_text: int = MOST_TEXT
) -> FileRead | Unread:
    match kind_of(name):
        case FileKind.DOCUMENT:
            match document(raw, name, convert(raw, name)):
                case Unread() as unread:
                    return unread
                case Scan() as scan:
                    return FileRead((), scans=(scan,))
                case found:
                    return FileRead((found,))
        case FileKind.MAIL:
            return _mail(raw, name, _Asked(title_from, convert, most_text), MAIL_DEPTH)
        case _:
            return FileRead((note(raw, name, title_from),))


def read_file(asked: ToRead, whole: ReadWhole = at_most, converting: Convert = convert) -> Outcome:
    try:
        raw = whole(asked.path, asked.most, asked.scanned)
    except OSError as error:
        return Refused(error.strerror or str(error), Again.NEXT_INDEX)
    if len(raw) > asked.most:
        return too_large(len(raw), asked.most)
    read = read_whole(raw, asked.path.name, asked.title_from, converting)
    if isinstance(read, Unread):
        return Refused(read.problem, Again.NEXT_RELEASE if read.next_release else Again.NEVER)
    return read
