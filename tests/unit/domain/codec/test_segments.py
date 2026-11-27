import json
from typing import cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.arrays import floats, ints
from sift.domain.codec.postings import MOST_INFLATED_BYTES, blocked, lengths_of, positions_of, ranges_of
from sift.domain.codec.segments import (
    GAP,
    MOST_TIMES,
    LongPostings,
    Postings,
    ShortPostings,
    segment,
    stream,
)
from sift.domain.codec.shape import MOST_BLOCK, Shape
from sift.domain.rank.bm25 import term_scores, unquantized

SMALL = Shape(range_shift=2, block=2, long=3, lengths_shift=3)
MOST_BYTES = 9
words = st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), max_size=6)
chunks = st.lists(st.tuples(words, words, words), min_size=1, max_size=25)


def decoded(held: Postings) -> tuple[list[int], list[int], list[int]]:
    parts = held.held.blocks if isinstance(held.held, LongPostings) else (held.held,)
    ids: list[int] = []
    counts: list[int] = []
    positions: list[int] = []
    for part in parts:
        found, tf = blocked(part.postings)
        ids += ints(found)
        counts += ints(tf)
        positions += ints(positions_of(part.positions, tf))
    return ids, counts, positions


@given(chunks, st.integers(0, 1 << 40), st.sampled_from([SMALL, Shape()]))
def test_a_segment_holds_where_each_term_occurs(columns: list[tuple[list[str], ...]], first: int, shape: Shape) -> None:
    ids = first + 3 * np.arange(len(columns), dtype=np.int64)
    listed = ints(ids)
    streams = [stream(*each) for each in columns]
    written = segment(ids, streams, average=4.0, shape=shape)
    want = {
        term: [
            (listed[n], [p for p, token in enumerate(s) if token == term]) for n, s in enumerate(streams) if term in s
        ]
        for term in {token for s in streams for token in s} - {GAP}
    }
    got = {held.term: decoded(held) for held in written.postings}
    assert got.keys() == want.keys()
    for term, (found, counts, positions) in got.items():
        assert found == [chunk for chunk, _ in want[term]]
        assert counts == [len(places) for _, places in want[term]]
        assert positions == [p for _, places in want[term] for p in places]
    assert {held.term: held.chunks for held in written.postings} == {term: len(each) for term, each in want.items()}
    lengths = {listed[n]: sum(token != GAP for token in s) for n, s in enumerate(streams)}
    assert written.tokens == sum(lengths.values())
    assert (written.first, written.last, written.chunks) == (listed[0], listed[-1], len(ids))
    for block, data in written.lengths.items():
        dense = lengths_of(data, shape.lengths_shift)
        for chunk, length in lengths.items():
            if chunk >> shape.lengths_shift == block:
                assert dense[chunk & ((1 << shape.lengths_shift) - 1)] == length


@given(chunks)
def test_the_ranges_of_a_long_list_bound_every_score_in_them(columns: list[tuple[list[str], ...]]) -> None:
    ids = np.arange(len(columns), dtype=np.int64) + 100
    streams = [stream(*each) for each in columns]
    lengths = np.array([sum(token != GAP for token in s) for s in streams])
    for held in segment(ids, streams, average=3.0, shape=SMALL).postings:
        if not isinstance(held.held, LongPostings):
            assert isinstance(held.held, ShortPostings)
            assert held.chunks < SMALL.long
            continue
        assert held.chunks >= SMALL.long
        found, counts, _ = decoded(held)
        scores = term_scores(np.array(counts), lengths[np.array(found) - 100], 3.0)
        ranges = ranges_of(held.held.ranges)
        bests = floats(unquantized(ranges.best))
        for rid, start, end, best in zip(ints(ranges.ids), ints(ranges.starts), ints(ranges.ends), bests, strict=True):
            assert all(chunk >> SMALL.range_shift == rid for chunk in found[start:end])
            assert best >= scores[start:end].max()
        assert ranges.ends[-1] == len(found)


@pytest.mark.parametrize(
    "asked, refusal",
    [
        ({"range_shift": 9}, "its counts are bytes"),
        ({"range_shift": -1}, "its counts are bytes"),
        ({"block": 0}, "one entry at least"),
        ({"lengths_shift": 0}, "one entry at least"),
        ({"block": (1 << 12) + 1}, "at most 4,096"),
        ({"long": (1 << 12) + 1}, "at most 4,096"),
        ({"lengths_shift": 21}, r"2\*\*20"),
    ],
)
def test_a_shape_whose_ranges_or_blocks_cannot_be_written_is_refused(asked: dict[str, int], refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        Shape(**asked)


def test_a_word_counts_in_a_passage_no_more_often_than_a_list_may_hold_and_the_passage_stays_as_long() -> None:
    often = MOST_TIMES + 500
    streams = [stream([], ["x"] * often, ["y"]), stream(["x"], [], [])]
    written = segment(np.array([7, 9], np.int64), streams, average=4.0)
    held = {each.term: decoded(each) for each in written.postings}
    assert held["x"] == ([7, 9], [MOST_TIMES, 1], [*range(1, MOST_TIMES + 1), 0])
    assert held["y"] == ([7], [1], [often + 2])
    assert written.tokens == often + 2


def test_no_list_a_shape_lets_be_written_unpacks_to_more_than_a_reader_takes() -> None:
    assert MOST_BLOCK * MOST_TIMES * MOST_BYTES <= MOST_INFLATED_BYTES


def test_a_chunk_of_no_word_is_in_the_segment_with_no_tokens() -> None:
    written = segment(np.array([7, 8], np.int64), [stream([], [], []), stream(["a"], [], [])], average=0.5)
    assert [(held.term, held.chunks) for held in written.postings] == [("a", 1)]
    assert ints(lengths_of(written.lengths[0], 10)[7:9]) == [0, 1]


@pytest.mark.parametrize("written", ['{"range_shift": 7.5}', '{"block": true}', '{"long": "9"}'])
def test_a_shape_of_other_than_whole_numbers_is_refused(written: str) -> None:
    with pytest.raises(TypeError, match="whole numbers"):
        Shape(**cast("dict[str, int]", json.loads(written)))
