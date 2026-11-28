import contextlib
import itertools
import re
import sqlite3
import tracemalloc
from collections.abc import Callable, Generator
from typing import NamedTuple

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.api.ids import DocumentId
from sift.domain.errors import DamagedError
from sift.storage.database import Reading, Writing, reading, rows, run, writing
from sift.storage.layout import LAYOUT
from sift.storage.texts import (
    BLOCK,
    BLOCKS,
    TextSpan,
    blocks,
    line_at,
    line_count,
    lines_between,
    put_text,
    spans_text,
    spans_text_at_once,
    text_of,
)
from tests.texts import runs_joined

DOCUMENT = DocumentId(7)
LINE = re.compile(r"[^\n]*\n|[^\n]+")


def _texts_table() -> str:
    return next(step for step in LAYOUT[0].steps if isinstance(step, str) and step.startswith("CREATE TABLE texts("))


TEXTS_TABLE = _texts_table()

runs = st.lists(
    st.tuples(st.sampled_from(["a", "é", "\N{GRINNING FACE}", "\n", " ", "\r"]), st.integers(1, BLOCK + 2)),
    max_size=8,
)
long_texts = runs.map(runs_joined)
cuts = st.lists(st.integers(1, 3 * BLOCK), max_size=4)


@contextlib.contextmanager
def held(text: str) -> Generator[Reading]:
    with contextlib.closing(sqlite3.connect(":memory:")) as connection:
        with writing(connection) as transaction:
            run(transaction, "CREATE TABLE documents(id INTEGER PRIMARY KEY) STRICT")
            run(transaction, TEXTS_TABLE)
            run(transaction, "INSERT INTO documents(id) VALUES (?)", (DOCUMENT,))
            put_text(transaction, DOCUMENT, (text,))
        with reading(connection) as transaction:
            yield transaction


def _pieces(text: str, at: list[int]) -> list[str]:
    edges = [0, *sorted(min(each, len(text)) for each in at), len(text)]
    return [text[start:end] for start, end in itertools.pairwise(edges)]


def test_the_block_size_is_part_of_the_store_s_layout() -> None:
    assert BLOCK == 65_536
    assert f"BETWEEN 1 AND {BLOCK}" in TEXTS_TABLE


@given(long_texts, cuts)
def test_a_text_is_cut_into_whole_blocks_numbered_with_the_line_they_start_in(text: str, at: list[int]) -> None:
    found = list(blocks(_pieces(text, at)))
    assert "".join(block for _, _, block in found) == text
    assert [number for number, _, _ in found] == list(range(len(found)))
    assert all(len(block) == BLOCK for _, _, block in found[:-1])
    assert all(0 < len(block) <= BLOCK for _, _, block in found)
    assert [line for _, line, _ in found] == [1 + text.count("\n", 0, number * BLOCK) for number, _, _ in found]
    assert found == list(blocks([text]))


def test_a_text_in_one_piece_is_cut_without_copying_what_is_left_of_it() -> None:
    text = "a" * (16 << 20)
    tracemalloc.start()
    try:
        cut = sum(len(block) for _, _, block in blocks((text,)))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert cut == len(text)
    assert peak < 4 * BLOCK


@given(long_texts, st.data())
def test_a_text_kept_in_blocks_reads_back_whole_in_spans_and_in_lines(text: str, data: st.DataObject) -> None:
    every = [found.group() for found in LINE.finditer(text)]
    with held(text) as transaction:
        assert text_of(transaction, DOCUMENT) == text
        assert line_count(transaction, DOCUMENT) == len(every)
        if text:
            starts = data.draw(st.lists(st.integers(0, len(text) - 1), min_size=1, max_size=5))
            spans = [TextSpan(DOCUMENT, start, data.draw(st.integers(1, len(text) - start))) for start in starts]
            assert spans_text(transaction, spans) == [text[span.start : span.start + span.length] for span in spans]
        first = data.draw(st.integers(1, len(every) + 2))
        last = data.draw(st.integers(first, first + 3))
        read = lines_between(transaction, DOCUMENT, first, last)
        assert (read.text, read.total) == ("".join(every[first - 1 : last]), len(every))
        if first <= len(every):
            assert line_at(transaction, DOCUMENT, first) == text.split("\n")[first - 1]


odd_texts = st.lists(
    st.tuples(st.sampled_from(["a", "\x01", "\x1f", '"', "\\", "\N{GRINNING FACE}", "\n"]), st.integers(1, BLOCK + 2)),
    min_size=1,
    max_size=6,
).map(runs_joined)


@given(odd_texts, st.data())
def test_spans_read_at_once_are_the_spans_read_row_by_row(text: str, data: st.DataObject) -> None:
    starts = data.draw(st.lists(st.integers(0, len(text) - 1), min_size=1, max_size=6))
    spans = [TextSpan(DOCUMENT, start, data.draw(st.integers(0, len(text) - start))) for start in starts]
    with held(text) as transaction:
        cut = [text[span.start : span.start + span.length] for span in spans]
        assert spans_text(transaction, spans) == cut
        assert spans_text_at_once(transaction, spans) == cut


@pytest.mark.parametrize("read", [spans_text, spans_text_at_once])
def test_a_chunk_reaching_past_its_text_is_damage(read: Callable[[Reading, list[TextSpan]], list[str]]) -> None:
    with held("a" * (BLOCK + 3)) as transaction:
        run(Writing(transaction.connection), "DELETE FROM texts WHERE block = 1")
        with pytest.raises(DamagedError, match="past the text of its document"):
            read(transaction, [TextSpan(DOCUMENT, BLOCK - 1, 3)])


def test_an_empty_text_has_no_block_and_no_line() -> None:
    with held("") as transaction:
        assert (text_of(transaction, DOCUMENT), line_count(transaction, DOCUMENT)) == ("", 0)
        assert lines_between(transaction, DOCUMENT, 1, 2).text == ""


class Step(NamedTuple):
    id: int
    parent: int
    unused: int
    detail: str


def test_a_passage_or_a_line_reads_only_the_blocks_it_lies_in() -> None:
    text = "".join(f"line {number}\n" for number in range(40_000))
    with contextlib.closing(sqlite3.connect(":memory:")) as connection:
        with writing(connection) as transaction:
            run(transaction, "CREATE TABLE documents(id INTEGER PRIMARY KEY) STRICT")
            run(transaction, TEXTS_TABLE)
            run(transaction, "INSERT INTO documents(id) VALUES (?)", (DOCUMENT,))
            put_text(transaction, DOCUMENT, (text,))
            middle = text.index("line 20000\n")
            kept = {middle // BLOCK, len(text) // BLOCK}
            run(transaction, "DELETE FROM texts WHERE block NOT IN (?, ?)", tuple(kept))
        with reading(connection) as transaction:
            assert spans_text(transaction, [TextSpan(DOCUMENT, middle, 10)]) == ["line 20000"]
            assert line_at(transaction, DOCUMENT, 20_001) == "line 20000"
            plan = rows(transaction, Step, f"EXPLAIN QUERY PLAN {BLOCKS}", ("[[7, 0]]",))
        assert any(step.detail.startswith("SEARCH t USING INDEX") for step in plan)
        assert not [step for step in plan if step.detail.startswith("SCAN t")]


RL = "# Reward shaping\n\nReward shaping speeds up learning.\n"


@pytest.mark.parametrize(
    "text, span, cut",
    [
        (RL, (1, 1), "# Reward shaping\n"),
        (RL, (2, 3), "\nReward shaping speeds up learning.\n"),
        (RL, (3, 99), RL[18:]),
        ("a\nb", (2, 2), "b"),
        ("a\N{LINE SEPARATOR}b\r\nc", (1, 1), "a\N{LINE SEPARATOR}b\r\n"),
    ],
)
def test_lines_are_cut_as_written_with_their_endings_and_only_a_newline_ends_one(
    text: str, span: tuple[int, int], cut: str
) -> None:
    with held(text) as transaction:
        assert lines_between(transaction, DOCUMENT, *span).text == cut
