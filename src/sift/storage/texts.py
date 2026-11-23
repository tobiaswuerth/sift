import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from sift.api.ids import DocumentId
from sift.domain.errors import DamagedError
from sift.storage.database import Reading, Writing, first, rows, run_each, streamed

BLOCK = 1 << 16
NEWLINE = "\n"
BLOCKS = """SELECT t.document, t.block, t.text
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


class _Whole(NamedTuple):
    id: DocumentId
    text: str


class _Lined(NamedTuple):
    first_line: int
    text: str


@dataclass(frozen=True, slots=True)
class LinesRead:
    text: str
    total: int


def blocks(pieces: Iterable[str]) -> Iterator[tuple[int, int, str]]:
    held, number, line = "", 0, 1
    for piece in pieces:
        held += piece
        while len(held) >= BLOCK:
            block, held = held[:BLOCK], held[BLOCK:]
            yield number, line, block
            number, line = number + 1, line + block.count(NEWLINE)
    if held:
        yield number, line, held


def put_text(transaction: Writing, document: DocumentId, pieces: Iterable[str]) -> None:
    run_each(
        transaction,
        "INSERT INTO texts(document, block, first_line, text) VALUES (?, ?, ?, ?)",
        ((document, number, line, text) for number, line, text in blocks(pieces)),
    )


def texts_cut(transaction: Writing) -> None:
    for found in streamed(transaction, _Whole, "SELECT id, text FROM documents WHERE text <> '' ORDER BY id", size=1):
        put_text(transaction, found.id, (found.text,))


def text_of(transaction: Reading, document: DocumentId) -> str:
    return "".join(found.text for found in rows(transaction, _Block, WHOLE, (document,)))


def _covering(span: TextSpan) -> range:
    return range(span.start // BLOCK, (span.start + max(span.length, 1) - 1) // BLOCK + 1)


def spans_text(transaction: Reading, spans: Sequence[TextSpan]) -> list[str]:
    wanted = sorted({(span.document, number) for span in spans for number in _covering(span)})
    held = {
        (found.document, found.block): found.text for found in rows(transaction, _Block, BLOCKS, (json.dumps(wanted),))
    }
    if len(held) < len(wanted):
        raise DamagedError("a chunk of the store reaches past the text of its document")
    cut: list[str] = []
    for span in spans:
        covering = _covering(span)
        at = span.start - covering.start * BLOCK
        cut.append("".join(held[span.document, number] for number in covering)[at : at + span.length])
    return cut


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
