import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import NamedTuple

HEADING = re.compile(r" {0,3}#{1,6}[ \t]")
FENCE = re.compile(r"((?:[ \t]*>)*[ \t]*)(`{3,}|~{3,})")
LIST_INDENT = 4
LINE_BREAKS = re.compile(r"[\r\n]+")
NEWLINE = "\n"
MOST_LINE = 1 << 20


class TextLine(NamedTuple):
    text: str
    number: int
    start: int
    whole: bool = True
    first: bool = True
    ends: bool = True


@dataclass(frozen=True, slots=True)
class Fence:
    mark: str | None = None
    behind: str = ""


OPEN = Fence()


def _beyond(behind: str, line: str) -> bool:
    if not line.strip():
        return False
    if ">" in behind:
        return not line.lstrip(" \t").startswith(">")
    indent = len(behind.expandtabs(LIST_INDENT))
    lead = line[: len(line) - len(line.lstrip(" \t"))]
    return indent >= LIST_INDENT and len(lead.expandtabs(LIST_INDENT)) < indent


def _fence(line: str, open_mark: str | None) -> re.Match[str] | None:
    found = FENCE.match(line)
    if found and not open_mark and found.group(2)[0] == "`" and "`" in line[found.end() :]:
        return None
    return found


def _closes(found: re.Match[str], line: str, open_mark: str) -> bool:
    mark = found.group(2)
    return mark[0] == open_mark[0] and len(mark) >= len(open_mark) and not line[found.end() :].strip()


def fenced(state: Fence, line: str) -> tuple[Fence, bool]:
    if state.mark and _beyond(state.behind, line):
        state = OPEN
    found = _fence(line, state.mark) if "```" in line or "~~~" in line else None
    if found is None:
        return state, not state.mark
    if not state.mark:
        return Fence(found.group(2), found.group(1)), False
    return (OPEN if _closes(found, line, state.mark) else state), False


def _cut(text: str, number: int, start: int, *, first: bool, most: int) -> Iterator[TextLine]:
    if first and len(text) <= most:
        yield TextLine(text, number, start)
        return
    for at in range(0, len(text), most):
        piece = text[at : at + most]
        yield TextLine(
            piece, number, start + at, whole=False, first=first and at == 0, ends=at + len(piece) == len(text)
        )


def text_lines(
    pieces: Iterable[str], first_line: int = 1, offset: int = 0, most: int = MOST_LINE
) -> Iterator[TextLine]:
    held: list[str] = []
    size, number, start, split = 0, first_line, offset, False
    for piece in pieces:
        at = 0
        while (end := piece.find(NEWLINE, at)) >= 0:
            if not held and end + 1 - at <= most:
                yield TextLine(piece[at : end + 1], number, start)
                number, start, at = number + 1, start + end + 1 - at, end + 1
                continue
            rest = "".join((*held, piece[at : end + 1]))
            yield from _cut(rest, number, start, first=not split, most=most)
            held, size, number, start, split, at = [], 0, number + 1, start + len(rest), False, end + 1
        if at < len(piece):
            held.append(piece[at:])
            size += len(piece) - at
        while size > most:
            joined = "".join(held)
            yield TextLine(joined[:most], number, start, whole=False, first=not split, ends=False)
            held, size, start, split = [joined[most:]], size - most, start + most, True
    if held:
        yield from _cut("".join(held), number, start, first=not split, most=most)


def with_prose(lines: Iterable[TextLine]) -> Iterator[tuple[TextLine, bool]]:
    state, outside = OPEN, True
    for line in lines:
        if line.first:
            state, outside = fenced(state, line.text)
        yield line, outside and line.whole


def _heading_text(line: str) -> str:
    title = line.strip().lstrip("#").strip()
    bare = title.rstrip("#")
    if bare != title and (not bare or bare[-1] in " \t"):
        return bare.strip()
    return title


def heading_of(line: str) -> str | None:
    return (_heading_text(line) or None) if HEADING.match(line) else None


def heading(text: str) -> str | None:
    lines = with_prose(text_lines((text,)))
    return next((title for line, outside in lines if outside and (title := heading_of(line.text))), None)


def one_line(title: str) -> str:
    return " ".join(part.strip(" \t") for part in LINE_BREAKS.split(title.strip()))
