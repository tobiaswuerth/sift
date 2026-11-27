import itertools
from collections.abc import Iterator, Sequence

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from sift.domain.codec.segments import (
    GAP,
    ListBlockRow,
    MergedChunks,
    MergeLimits,
    Merger,
    MergeStep,
    WrittenSegment,
    fed,
    merger,
    stepped,
    stream,
)
from sift.domain.codec.shape import Shape
from sift.domain.errors import DamagedError
from tests.unit.domain.codec.built import SEGMENT, SMALL, Columns, chunks, of_bodies, segment

EVERYTHING = 10**9
WHOLE = MergeLimits(postings=EVERYTHING, stored=EVERYTHING, tokens=EVERYTHING)
PIECES = MergeLimits(postings=1, stored=1, tokens=EVERYTHING)
DENSE = MergeLimits(postings=EVERYTHING, stored=EVERYTHING, tokens=1)
AVERAGE = 4.0

type Source = list[tuple[str, int, bytes, bytes]]


def sources(made: WrittenSegment, shape: Shape) -> list[Source]:
    short: Source = [
        (term, held, coded, placed)
        for _, term, held, coded, placed, _ in made.lists
        if coded is not None and placed is not None
    ]
    blocks: Source = [(term, shape.block, coded, placed) for _, term, _, coded, placed in made.blocks]
    return [short, blocks]


def steps(held: Merger, given: Sequence[Source], at_once: int) -> Iterator[tuple[MergeStep, int]]:
    left = [itertools.batched(each, at_once) for each in given]
    taken, done = 0, False
    while not done:
        step = stepped(held, SEGMENT)
        yield step, taken
        for at in step.wanted:
            more = True
            while more:
                rows = list(next(left[at], ()))
                more = fed(held, at, rows)
                taken += len(rows)
        done = step.done


def whole(ids: Sequence[int]) -> NDArray[np.int64]:
    return np.array(ids, np.int64)


def numbered(row: ListBlockRow) -> tuple[str, int]:
    return row[1], row[2]


@given(
    st.lists(chunks, min_size=1, max_size=4),
    st.data(),
    st.sampled_from([SMALL, Shape()]),
    st.sampled_from([WHOLE, PIECES, DENSE]),
    st.sampled_from([1, 2, 64]),
)
def test_segments_merge_into_the_segment_their_live_chunks_alone_build(
    parts: list[list[Columns]], data: st.DataObject, shape: Shape, limits: MergeLimits, at_once: int
) -> None:
    columns = [chunk for part in parts for chunk in part]
    ids = 5 + 3 * np.arange(len(columns), dtype=np.int64)
    dead = sorted(data.draw(st.sets(st.integers(1, len(columns) - 1))) if len(columns) > 1 else set[int]())
    live = [n for n in range(len(columns)) if n not in dead]
    given = [
        source
        for start, end in itertools.pairwise(itertools.accumulate(map(len, parts), initial=0))
        for source in sources(segment(ids[start:end], columns[start:end], AVERAGE, shape), shape)
    ]
    lengths = whole([sum(token != GAP for token in stream(*columns[n])) for n in live])
    held = merger(MergedChunks(ids[whole(live)], lengths, ids[whole(dead)], AVERAGE), shape, limits, len(given))
    made = [step for step, _ in steps(held, given, at_once)]
    want = segment(ids[whole(live)], [columns[n] for n in live], AVERAGE, shape)
    assert [row for step in made for row in step.lists] == list(want.lists)
    assert sorted((row for step in made for row in step.blocks), key=numbered) == list(want.blocks)


def thrice(words: str, first: int) -> list[Source]:
    return sources(segment(first + np.arange(3, dtype=np.int64), of_bodies([words.split()] * 3), AVERAGE), Shape())


def chunks_of(ids: range) -> MergedChunks:
    return MergedChunks(whole(ids), whole([2] * len(ids)), whole([]), AVERAGE)


def test_a_merge_takes_terms_together_up_to_its_postings_or_bytes_and_a_term_past_either_alone() -> None:
    given = [*thrice("a b", 0), *thrice("a c", 3)]
    (size,) = {len(coded) + len(placed) for source in given for _, _, coded, placed in source}

    def taken(limits: MergeLimits) -> list[list[str]]:
        held = merger(chunks_of(range(6)), Shape(), limits, len(given))
        for at, rows in enumerate(given):
            fed(held, at, rows)
            fed(held, at, [])
        made = [[term for _, term, *_ in step.lists] for step, _ in steps(held, given, 1)]
        return [terms for terms in made if terms]

    assert taken(MergeLimits(EVERYTHING, 1, EVERYTHING)) == [["a"], ["b"], ["c"]]
    assert taken(MergeLimits(4, EVERYTHING, EVERYTHING)) == [["a"], ["b", "c"]]
    assert taken(MergeLimits(EVERYTHING, 3 * size, EVERYTHING)) == [["a", "b"], ["c"]]
    assert taken(WHOLE) == [["a", "b", "c"]]


def word(number: int) -> str:
    return "".join("abcdefghij"[int(digit)] for digit in f"{number:05}")


def test_a_merge_asks_for_the_rows_of_a_source_once_it_holds_few_of_them_however_many_there_are() -> None:
    at_once, terms, a_chunk = 512, 20_000, 100
    given = [
        source
        for start in (0, 1)
        for source in sources(
            segment(
                start * terms + np.arange(terms // a_chunk, dtype=np.int64),
                of_bodies([word(2 * n + start) for n in range(at, at + a_chunk)] for at in range(0, terms, a_chunk)),
                AVERAGE,
            ),
            Shape(),
        )
    ]
    ids = whole([*range(terms // a_chunk), *range(terms, terms + terms // a_chunk)])
    ahead = 4096
    limits = MergeLimits(len(given) * ahead, EVERYTHING, EVERYTHING)
    held = merger(MergedChunks(ids, whole([a_chunk] * len(ids)), whole([]), AVERAGE), Shape(), limits, len(given))
    most = 2 * (ahead + at_once)
    assert 2 * terms > 4 * most
    written, deepest = 0, 0
    for step, taken in steps(held, given, at_once):
        deepest = max(deepest, taken - written)
        written += len(step.lists)
    assert 2 * ahead <= deepest <= most
    assert written == 2 * terms


def test_a_term_past_what_a_merge_takes_at_once_comes_back_in_blocks_as_its_rows_are_fed() -> None:
    a_segment = 40
    given = [
        source
        for first in (0, a_segment)
        for source in sources(
            segment(first + np.arange(a_segment, dtype=np.int64), of_bodies([["w"]] * a_segment), AVERAGE, SMALL), SMALL
        )
    ]
    held = merger(chunks_of(range(2 * a_segment)), SMALL, PIECES, len(given))
    made = [(len(step.blocks), [term for _, term, *_ in step.lists], taken) for step, taken in steps(held, given, 1)]
    written = list(itertools.accumulate(blocks for blocks, _, _ in made))
    assert all(taken - before <= 2 for (_, _, taken), before in zip(made, written, strict=True))
    assert [terms for _, terms, _ in made if terms] == [["w"]]
    assert written[[terms for _, terms, _ in made].index(["w"])] == written[-1] == a_segment


def test_a_merge_of_lists_that_are_damaged_or_hold_what_the_store_does_not_is_damage() -> None:
    given = thrice("a", 0)
    held = merger(chunks_of(range(1, 3)), Shape(), WHOLE, len(given))
    with pytest.raises(DamagedError, match="chunks the store does not"):
        list(steps(held, given, 1))
    held = merger(chunks_of(range(3)), Shape(), WHOLE, len(given))
    fed(held, 0, [("a", 3, b"\x00", b"\x00")])
    fed(held, 0, [])
    fed(held, 1, [])
    with pytest.raises(DamagedError, match="shorter than its head"):
        stepped(held, SEGMENT)
    damaged: list[tuple[object, ...]] = [
        (b"a", 3, b"", b""),
        ("a", 3, None, b""),
        ("a", "3", b"", b""),
        ("a", -1, b"", b""),
    ]
    for row in damaged:
        with pytest.raises(DamagedError, match="is no list of a term"):
            fed(held, 0, [row])


def test_a_merge_takes_rows_of_the_sources_it_was_made_for_alone() -> None:
    held = merger(chunks_of(range(3)), Shape(), WHOLE, 2)
    with pytest.raises(IndexError, match="a source it has not"):
        fed(held, 2, [("a", 3, b"", b"")])
