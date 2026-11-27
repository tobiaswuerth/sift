import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from sift.domain.text.markdown import HEADING, TextLine, text_lines, with_prose

MOST_CHARACTERS = 3000
MOST_OVERLAP = 300
HEADING_MARK = "#"


@dataclass(frozen=True, slots=True)
class Chunk:
    first_line: int
    last_line: int
    start: int
    length: int


class Piece(NamedTuple):
    number: int
    start: int
    text: str
    heading: bool


def cuts(line: str, *, ends: bool = True) -> Iterator[tuple[int, int]]:
    cut = 0
    while len(line) - cut > MOST_CHARACTERS:
        space = line.rfind(" ", cut, cut + MOST_CHARACTERS)
        end = space + 1 if space >= 0 else cut + MOST_CHARACTERS
        yield cut, end
        cut = end
    if ends:
        yield cut, len(line)


def pieces(lines: Iterable[tuple[TextLine, bool]]) -> Iterator[Piece]:
    carry, carry_start = "", 0
    for line, prose in lines:
        heading = prose and "#" in line.text and HEADING.match(line.text) is not None
        if not carry and line.ends and len(line.text) <= MOST_CHARACTERS:
            yield Piece(line.number, line.start, line.text, heading)
            continue
        held, start = (carry + line.text, carry_start) if carry else (line.text, line.start)
        done = 0
        for cut, end in cuts(held, ends=line.ends):
            yield Piece(line.number, start + cut, held[cut:end], heading and cut == 0)
            done = end
        carry, carry_start = ("", 0) if line.ends else (held[done:], start + done)


def _overlap(previous: Sequence[Piece], following: Piece) -> list[Piece]:
    room = min(MOST_OVERLAP, MOST_CHARACTERS - len(following.text))
    sizes = itertools.accumulate(len(piece.text) for piece in reversed(previous))
    carried = sum(1 for _ in itertools.takewhile(lambda size: size <= room, sizes))
    return list(previous[len(previous) - carried :])


def groups(lines: Iterator[Piece]) -> Iterator[list[Piece]]:
    group: list[Piece] = []
    size = 0
    for piece in lines:
        length = len(piece.text)
        if group and (size + length > MOST_CHARACTERS or (piece.heading and size > MOST_CHARACTERS // 2)):
            yield group
            group = [] if piece.heading else _overlap(group, piece)
            size = sum(len(carried.text) for carried in group)
        group.append(piece)
        size += length
    yield group


def chunk_of(group: Sequence[Piece]) -> Chunk | None:
    body = "".join(piece.text for piece in group)
    stripped = body.strip()
    if not stripped:
        return None
    lead = len(body) - len(body.lstrip())
    first = group[0].number + body.count("\n", 0, lead)
    return Chunk(first, first + stripped.count("\n"), group[0].start + lead, len(stripped))


def chunked(lines: Iterable[tuple[TextLine, bool]]) -> Iterator[Chunk]:
    for group in groups(pieces(lines)):
        if (found := chunk_of(group)) is not None:
            yield found


def _one(text: str, first_line: int, offset: int) -> tuple[Chunk, ...]:
    stripped = text.strip()
    if not stripped:
        return ()
    lead = len(text) - len(text.lstrip())
    first = first_line + text.count("\n", 0, lead)
    return (Chunk(first, first + stripped.count("\n"), offset + lead, len(stripped)),)


def chunks(text: str, first_line: int = 1, offset: int = 0) -> tuple[Chunk, ...]:
    if len(text) <= MOST_CHARACTERS and text.find(HEADING_MARK, MOST_CHARACTERS // 2) < 0:
        return _one(text, first_line, offset)
    return tuple(chunked(with_prose(text_lines((text,), first_line, offset))))
