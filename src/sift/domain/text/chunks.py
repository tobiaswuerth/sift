import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from sift.domain.text.markdown import HEADING, LINE, outside_code

MOST_CHARACTERS = 3000
MOST_OVERLAP = 300


@dataclass(frozen=True, slots=True)
class Chunk:
    first_line: int
    last_line: int
    start: int
    length: int


class Piece(NamedTuple):
    number: int
    start: int
    length: int
    heading: bool


def cuts(line: str) -> Iterator[tuple[int, int]]:
    cut = 0
    while len(line) - cut > MOST_CHARACTERS:
        space = line.rfind(" ", cut, cut + MOST_CHARACTERS)
        end = space + 1 if space >= 0 else cut + MOST_CHARACTERS
        yield cut, end
        cut = end
    yield cut, len(line)


def pieces(text: str, first_line: int) -> Iterator[Piece]:
    lines: list[str] = LINE.findall(text)
    start = 0
    for number, (line, prose) in enumerate(zip(lines, outside_code(lines), strict=True), first_line):
        heading = prose and "#" in line and HEADING.match(line) is not None
        if len(line) <= MOST_CHARACTERS:
            yield Piece(number, start, len(line), heading)
        else:
            for cut, end in cuts(line):
                yield Piece(number, start + cut, end - cut, heading and cut == 0)
        start += len(line)


def _overlap(previous: Sequence[Piece], following: Piece) -> list[Piece]:
    room = min(MOST_OVERLAP, MOST_CHARACTERS - following.length)
    sizes = itertools.accumulate(piece.length for piece in reversed(previous))
    carried = sum(1 for _ in itertools.takewhile(lambda size: size <= room, sizes))
    return list(previous[len(previous) - carried :])


def groups(lines: Iterator[Piece]) -> Iterator[list[Piece]]:
    group: list[Piece] = []
    size = 0
    for piece in lines:
        if group and (size + piece.length > MOST_CHARACTERS or (piece.heading and size > MOST_CHARACTERS // 2)):
            yield group
            group = [] if piece.heading else _overlap(group, piece)
            size = sum(carried.length for carried in group)
        group.append(piece)
        size += piece.length
    yield group


def chunks(text: str, first_line: int = 1, offset: int = 0) -> tuple[Chunk, ...]:
    found: list[Chunk] = []
    for group in groups(pieces(text, first_line)):
        if not group:
            continue
        start, end = group[0].start, group[-1].start + group[-1].length
        body = text[start:end]
        if stripped := body.strip():
            lead = len(body) - len(body.lstrip())
            first = group[0].number + body.count("\n", 0, lead)
            found.append(Chunk(first, first + stripped.count("\n"), offset + start + lead, len(stripped)))
    return tuple(found)
