import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from sift.domain.arrays import ints
from sift.domain.codec.postings import (
    PACKED,
    RAW,
    blocked,
    blocked_all,
    encoded,
    lengths_of,
    numbers,
    positions_of,
    ranged,
    ranges_of,
    varints,
    with_lengths,
)


def ascending(gaps: list[int], start: int) -> NDArray[np.int64]:
    return start + np.cumsum(np.array(gaps, np.int64))


type Listed = list[tuple[int, list[int]]]

posting = st.tuples(
    st.integers(1, (1 << 32) - 1), st.lists(st.integers(0, 1 << 30), min_size=1, max_size=8, unique=True)
)
lists = st.lists(st.lists(posting, min_size=1, max_size=40), min_size=1, max_size=12)


@given(lists, st.integers(0, 1 << 48))
def test_lists_encoded_together_each_come_back_with_their_counts_and_positions(each: list[Listed], start: int) -> None:
    ids: list[int] = []
    counts: list[int] = []
    positions: list[int] = []
    bounds = [0]
    for held in each:
        place = start
        for gap, where in held:
            place += gap
            ids.append(place)
            counts.append(len(where))
            positions += sorted(where)
        bounds.append(len(ids))
    found = encoded(np.array(ids, np.int64), np.array(counts, np.int64), np.array(positions, np.int64), bounds)
    together, together_counts, together_bounds = blocked_all(found.postings)
    assert (ints(together), ints(together_counts), together_bounds) == (ids, counts, bounds)
    at = 0
    for number, (low, high) in enumerate(itertools.pairwise(bounds)):
        back, tf = blocked(found.postings[number])
        assert ints(back) == ids[low:high]
        assert ints(tf) == counts[low:high]
        size = sum(counts[low:high])
        assert ints(positions_of(found.positions[number], tf)) == positions[at : at + size]
        at += size


def test_a_short_list_is_kept_as_it_is_and_a_long_one_packed() -> None:
    short = encoded(np.array([5, 9], np.int64), np.array([1, 2], np.int64), np.array([0, 1, 7], np.int64), [0, 2])
    assert short.postings[0][:1] == RAW
    ids = np.arange(0, 3000, 3, dtype=np.int64)
    long = encoded(ids, np.ones(len(ids), np.int64), np.zeros(len(ids), np.int64), [0, len(ids)])
    assert long.postings[0][:1] == PACKED
    assert len(long.postings[0]) < 100


def test_no_lists_decode_to_nothing() -> None:
    places, counts, bounds = blocked_all([])
    assert (ints(places), ints(counts), bounds) == ([], [], [0])


def test_ids_too_far_apart_for_a_block_are_refused() -> None:
    with pytest.raises(ValueError, match=r"2\*\*32 apart"):
        encoded(np.array([0, 1 << 32], np.int64), np.ones(2, np.int64), np.zeros(2, np.int64), [0, 2])


def test_ids_of_separate_lists_may_go_down_between_them() -> None:
    found = encoded(np.array([9, 3], np.int64), np.ones(2, np.int64), np.zeros(2, np.int64), [0, 1, 2])
    assert [ints(blocked(each)[0]) for each in found.postings] == [[9], [3]]


@given(st.lists(st.integers(0, (1 << 62) - 1)))
def test_numbers_come_back_from_their_varints(values: list[int]) -> None:
    written, offsets = varints(np.array(values, np.int64))
    assert ints(numbers(written)) == values
    assert offsets[-1] == len(written)


@pytest.mark.parametrize("value, size", [(0, 1), (127, 1), (128, 2), ((1 << 14) - 1, 2), (1 << 14, 3)])
def test_a_varint_takes_a_byte_per_seven_bits(value: int, size: int) -> None:
    assert len(varints(np.array([value], np.int64))[0]) == size


@given(st.lists(st.integers(1, 128), min_size=1, max_size=100), st.integers(0, 1 << 40), st.data())
def test_ranges_hold_where_their_postings_start_and_end_and_their_best_scores(
    counts: list[int], first: int, data: st.DataObject
) -> None:
    found = ascending(data.draw(st.lists(st.integers(1, 1000), min_size=len(counts), max_size=len(counts))), first)
    best = np.array(data.draw(st.lists(st.integers(0, 255), min_size=len(counts), max_size=len(counts))), np.uint8)
    back = ranges_of(ranged(found, np.array(counts, np.int64), best))
    assert ints(back.ids) == ints(found)
    assert ints(back.ends - back.starts) == counts
    assert ints(back.starts) == [0, *ints(back.ends[:-1])]
    assert ints(back.best) == ints(best)


@given(st.lists(st.integers(0, 5000), min_size=1, max_size=300, unique=True), st.data())
def test_lengths_come_back_by_id_in_blocks_of_consecutive_ids(chosen: list[int], data: st.DataObject) -> None:
    found = np.array(sorted(chosen), np.int64)
    drawn = data.draw(st.lists(st.integers(1, 1 << 20), min_size=len(found), max_size=len(found)))
    lengths = np.array(drawn, np.int64)
    written = with_lengths(found, lengths, shift=6)
    assert set(written) == set(ints(found >> 6))
    for number, held in written.items():
        dense = lengths_of(held)
        inside = np.flatnonzero(np.isin(found >> 6, [number]))
        assert ints(dense[found[inside] & 63]) == ints(lengths[inside])
        assert int(np.count_nonzero(dense)) == len(inside)
