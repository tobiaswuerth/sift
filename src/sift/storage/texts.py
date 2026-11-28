import json
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import NamedTuple, cast

from sift.api.ids import DocumentId
from sift.domain.errors import DamagedError
from sift.storage.database import Reading, Value, Writing, first, rows, run_each, streamed, touched

BLOCK = 1 << 16
NEWLINE = "\n"
BLOCKS = """SELECT t.document, t.block, t.text
            FROM json_each(?) j JOIN texts t
              ON t.document = json_extract(j.value, '$[0]') AND t.block = json_extract(j.value, '$[1]')"""
TOUCH_BLOCKS = """SELECT sum(length(t.text))
                  FROM json_each(?) j JOIN texts t
                    ON t.document = json_extract(j.value, '$[0]') AND t.block = json_extract(j.value, '$[1]')"""
BLOCKS_AT_ONCE = """SELECT json_group_array(json_array(t.document, t.block, t.text))
                    FROM json_each(?) j JOIN texts t
                      ON t.document = json_extract(j.value, '$[0]') AND t.block = json_extract(j.value, '$[1]')"""
WHOLE = "SELECT document, block, text FROM texts WHERE document = ? ORDER BY block"
FROM_LINE = """SELECT first_line, text FROM texts
               WHERE document = ? AND block >= coalesce(
                   (SELECT max(block) FROM texts WHERE document = ? AND first_line < ?), 0)
               ORDER BY block"""
LAST = "SELECT first_line, text FROM texts WHERE document = ? ORDER BY block DESC LIMIT 1"


class TextSpan(NamedTuple):
    document: DocumentId
    start: int
    length: int


class _Block(NamedTuple):
    document: DocumentId
    block: int
    text: str


class _Lined(NamedTuple):
    first_line: int
    text: str


@dataclass(frozen=True, slots=True)
class LinesRead:
    text: str
    total: int


def blocks(pieces: Iterable[str]) -> Iterator[tuple[int, int, str]]:
    held: list[str] = []
    size, number, line = 0, 0, 1
    for piece in pieces:
        at = 0
        while len(piece) - at >= BLOCK - size:
            end = at + BLOCK - size
            block = "".join((*held, piece[at:end]))
            yield number, line, block
            held, size, at, number, line = [], 0, end, number + 1, line + block.count(NEWLINE)
        if at < len(piece):
            held.append(piece[at:])
            size += len(piece) - at
    if held:
        yield number, line, "".join(held)


def put_texts(transaction: Writing, texts: Iterable[tuple[DocumentId, Iterable[str]]]) -> None:
    run_each(
        transaction,
        "INSERT INTO texts(document, block, first_line, text) VALUES (?, ?, ?, ?)",
        ((document, number, line, text) for document, pieces in texts for number, line, text in blocks(pieces)),
    )


def put_text(transaction: Writing, document: DocumentId, pieces: Iterable[str]) -> None:
    put_texts(transaction, [(document, pieces)])


def text_of(transaction: Reading, document: DocumentId) -> str:
    return "".join(found.text for found in rows(transaction, _Block, WHOLE, (document,)))


def _covering(span: TextSpan) -> range:
    return range(span.start // BLOCK, (span.start + max(span.length, 1) - 1) // BLOCK + 1)


def _wanted(spans: Sequence[TextSpan]) -> list[tuple[DocumentId, int]]:
    return sorted({(span.document, number) for span in spans for number in _covering(span)})


def _cut(spans: Sequence[TextSpan], wanted: Sized, held: Mapping[tuple[int, int], str]) -> list[str]:
    if len(held) < len(wanted):
        raise DamagedError("a chunk of the store reaches past the text of its document")
    cut: list[str] = []
    for span in spans:
        covering = _covering(span)
        at = span.start - covering.start * BLOCK
        cut.append("".join(held[span.document, number] for number in covering)[at : at + span.length])
    return cut


def spans_text(transaction: Reading, spans: Sequence[TextSpan]) -> list[str]:
    wanted = _wanted(spans)
    touched(transaction, TOUCH_BLOCKS, wanted)
    found = rows(transaction, _Block, BLOCKS, (json.dumps(wanted),))
    return _cut(spans, wanted, {(each.document, each.block): each.text for each in found})


def spans_text_at_once(transaction: Reading, spans: Sequence[TextSpan]) -> list[str]:
    wanted = _wanted(spans)
    (found,) = rows(transaction, Value, BLOCKS_AT_ONCE, (json.dumps(wanted),))
    read = cast("list[tuple[int, int, str]]", json.loads(found.value))
    return _cut(spans, wanted, {(document, block): text for document, block, text in read})


def line_count(transaction: Reading, document: DocumentId) -> int:
    last = first(transaction, _Lined, LAST, (document,))
    if last is None:
        return 0
    return last.first_line - 1 + last.text.count(NEWLINE) + (not last.text.endswith(NEWLINE))


def _passed(text: str, at: int, newlines: int) -> tuple[int, int]:
    while newlines and (found := text.find(NEWLINE, at)) >= 0:
        at, newlines = found + 1, newlines - 1
    return at, newlines


def lines_between(transaction: Reading, document: DocumentId, first_line: int, last_line: int) -> LinesRead:
    total = line_count(transaction, document)
    if first_line > total:
        return LinesRead("", total)
    held: list[str] = []
    wanted, at = last_line - first_line + 1, None
    for block in streamed(transaction, _Lined, FROM_LINE, (document, document, first_line), size=1):
        at = 0 if at is not None else _passed(block.text, 0, first_line - block.first_line)[0]
        end, wanted = _passed(block.text, at, wanted)
        held.append(block.text[at : end if not wanted else len(block.text)])
        if not wanted:
            break
    return LinesRead("".join(held), total)


def line_at(transaction: Reading, document: DocumentId, number: int) -> str:
    return lines_between(transaction, document, number, number).text.removesuffix(NEWLINE)
