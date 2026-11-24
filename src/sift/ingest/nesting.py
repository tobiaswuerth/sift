import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from email import policy
from email.message import EmailMessage
from email.parser import BytesHeaderParser

from sift.domain.text.heads import refuse_unreadable

LINE_END = re.compile(rb"\r\n|\r|\n")
MARKED_LINE_START = re.compile(rb"(?:\r\n|\r(?!\n)|\n)(?=--)")
MARKED_OR_BLANK_LINE_START = re.compile(rb"(?:\r\n|\r(?!\n)|\n)(?=--|\r|\n)")
HEADER_LINE = re.compile(rb"From |[\x21-\x39\x3b-\x7e]*:|[\t ]")
BOUNDARY_END = re.compile(rb"(--)?[ \t]*")
AS_READ = re.compile("[\x00-\x7f\udc80-\udcff]*")
DASHES = b"--"
LINE_ENDS = (b"\r", b"\n")
ATTACHED_DEFAULT = "message/rfc822"
STATUS = "message/delivery-status"


@dataclass(frozen=True, slots=True)
class _Open:
    dashes: bytes | None = None
    attached: bool = False
    digest: bool = False
    started: bool = False
    status: bool = False


def _line(raw: bytes, at: int) -> tuple[bytes, int]:
    found = LINE_END.search(raw, at)
    return (raw[at : found.start()], found.end()) if found else (raw[at:], len(raw))


def _closes(line: bytes, part: _Open) -> bool | None:
    if part.dashes is None or not line.startswith(part.dashes):
        return None
    found = BOUNDARY_END.fullmatch(line, len(part.dashes))
    return None if found is None else found.group(1) is not None


def _ended_by(line: bytes, opened: Sequence[_Open]) -> tuple[int, bool] | None:
    for depth, part in enumerate(opened):
        if part.started and (closing := _closes(line, part)) is not None:
            return depth, closing
    if opened and not opened[-1].started and (closing := _closes(line, opened[-1])) is not None:
        return len(opened) - 1, closing
    return None


def _status(opened: Sequence[_Open]) -> int | None:
    return next((depth for depth, part in enumerate(opened) if part.status), None)


def _head(raw: bytes, at: int, opened: Sequence[_Open]) -> tuple[bytes, int, bool]:
    start = at
    while at < len(raw):
        line, after = _line(raw, at)
        if not line:
            return raw[start:at], after, True
        if _ended_by(line, opened) is not None or not HEADER_LINE.match(line):
            break
        at = after
    return raw[start:at], at, False


def _declared(head: bytes, *, in_digest: bool) -> _Open | None:
    headers = BytesHeaderParser(EmailMessage, policy=policy.default).parsebytes(head)
    if in_digest:
        headers.set_default_type(ATTACHED_DEFAULT)
    if headers.get_content_type() == STATUS:
        return _Open(status=True)
    kind = headers.get_content_maintype()
    if kind == "message":
        return _Open(attached=True)
    boundary = headers.get_boundary() if kind == "multipart" else None
    if boundary is None:
        return None
    dashes = DASHES + boundary.encode("ascii", "surrogateescape") if AS_READ.fullmatch(boundary) else None
    return _Open(dashes, digest=headers.get_content_subtype() == "digest")


def _next_mark(raw: bytes, at: int, *, blank: bool) -> int | None:
    if raw.startswith(DASHES, at) or (blank and raw[at : at + 1] in LINE_ENDS):
        return at
    found = (MARKED_OR_BLANK_LINE_START if blank else MARKED_LINE_START).search(raw, at)
    return found.end() if found else None


def _past_boundaries(raw: bytes, at: int, opened: Sequence[_Open]) -> int:
    *outer, part = opened
    while at < len(raw):
        line, after = _line(raw, at)
        if _ended_by(line, outer) is not None or _closes(line, part) is None:
            return at
        at = after
    return at


def nesting(raw: bytes, most: int) -> int:
    opened: list[_Open] = []
    deepest, at, heading = 0, 0, True
    while heading or at < len(raw):
        status = _status(opened)
        if heading:
            head, at, blank = _head(raw, at, opened)
            refuse_unreadable(head)
            found = _declared(head, in_digest=bool(opened) and opened[-1].digest)
            if found is not None:
                opened.append(found)
                deepest = max(deepest, len(opened))
            if deepest > most:
                return deepest
            if blank and status is not None:
                del opened[status + 1 :]
                heading = at < len(raw)
                continue
            heading = found is not None and (found.attached or found.status) and at < len(raw)
            continue
        start = _next_mark(raw, at, blank=status is not None)
        if start is None:
            return deepest
        line, at = _line(raw, start)
        if status is not None and not line:
            del opened[status + 1 :]
            heading = True
            continue
        if (ended := _ended_by(line, opened)) is None:
            continue
        depth, closing = ended
        del opened[depth + (0 if closing else 1) :]
        if not closing:
            opened[depth] = replace(opened[depth], started=True)
            at = _past_boundaries(raw, at, opened)
        heading = not closing
    return deepest
