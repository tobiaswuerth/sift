import posixpath
from dataclasses import replace

from sift.api.collections import TitleFrom
from sift.domain.text.documents import Convert, FileRead, Unread, document
from sift.domain.text.kinds import FileKind, kind_of
from sift.domain.text.links import folded
from sift.domain.text.notes import Document, note
from sift.ingest.mail import Attachment, mail

MAIL_DEPTH = 3


def unique(name: str, taken: set[str]) -> str:
    stem, extension = posixpath.splitext(name)
    key, number = name, 1
    while folded(key) in taken:
        number += 1
        key = f"{stem} ({number}){extension}"
    taken.add(folded(key))
    return key


def within(key: str, inner: str) -> str:
    return f"{key}/{inner}" if inner else key


def _attachment(attached: Attachment, title_from: TitleFrom, convert: Convert, depth: int) -> FileRead | Unread | None:
    if not attached.mail:
        return read_whole(attached.raw, attached.name, title_from, convert)
    return _mail(attached.raw, attached.name, title_from, convert, depth - 1) if depth else None


def _mail(raw: bytes, name: str, title_from: TitleFrom, convert: Convert, depth: int) -> FileRead | Unread:
    found = mail(raw, name, convert)
    if isinstance(found, Unread):
        return found
    documents: list[Document] = [found.document]
    problems = list(found.problems)
    taken: set[str] = set()
    for attached in found.attachments:
        read = _attachment(attached, title_from, convert, depth)
        if read is None:
            continue
        key = unique(attached.name, taken)
        match read:
            case Unread(problem=problem):
                problems.append(f"{attached.name}: {problem}")
            case _:
                documents += [replace(each, key=within(key, each.key)) for each in read.documents]
                problems += [f"{attached.name}: {problem}" for problem in read.problems]
    return FileRead(tuple(documents), tuple(problems))


def read_whole(raw: bytes, name: str, title_from: TitleFrom, convert: Convert) -> FileRead | Unread:
    match kind_of(name):
        case FileKind.DOCUMENT:
            found = document(raw, name, convert(raw, name))
            return found if isinstance(found, Unread) else FileRead((found,))
        case FileKind.MAIL:
            return _mail(raw, name, title_from, convert, MAIL_DEPTH)
        case _:
            return FileRead((note(raw, name, title_from),))
