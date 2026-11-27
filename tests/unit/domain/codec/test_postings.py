import contextlib
import itertools
import tracemalloc
import zlib

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from sift.domain.arrays import ints
from sift.domain.codec.postings import (
    MOST_INFLATED_BYTES,
    PACKED,
    RANGES_HEAD,
    RAW,
    blocked,
    blocked_all,
    lengths_of,
    positions_of,
    ranges_of,
    with_lengths,
)
from sift.domain.codec.shape import Shape
from sift.domain.errors import DamagedError
from tests.coded_lists import BLOCK_HEAD, block, placed, ranges, varint, varints
from tests.unit.domain.codec.built import SMALL, of_bodies, segment, short_lists


def ascending(gaps: list[int], start: int) -> NDArray[np.int64]:
    return start + np.cumsum(np.array(gaps, np.int64))


type Listed = list[tuple[int, list[int]]]

posting = st.tuples(
    st.integers(1, (1 << 32) - 1), st.lists(st.integers(0, 1 << 30), min_size=1, max_size=8, unique=True)
)
lists = st.lists(st.lists(posting, min_size=1, max_size=40), min_size=1, max_size=12)


@given(lists, st.integers(0, 1 << 48))
def test_lists_each_come_back_with_their_counts_and_positions_alone_or_together(each: list[Listed], start: int) -> None:
    ids: list[list[int]] = []
    positions: list[list[list[int]]] = []
    for held in each:
        ids.append([*itertools.accumulate((gap for gap, _ in held), initial=start)][1:])
        positions.append([sorted(where) for _, where in held])
    counts = [[len(where) for where in one] for one in positions]
    coded = [block(found, tf) for found, tf in zip(ids, counts, strict=True)]
    together, together_counts, bounds = blocked_all(coded)
    assert ints(together) == [place for one in ids for place in one]
    assert ints(together_counts) == [count for one in counts for count in one]
    assert bounds == [*itertools.accumulate(map(len, ids), initial=0)]
    for number, found in enumerate(ids):
        back, tf = blocked(coded[number])
        assert (ints(back), ints(tf)) == (found, counts[number])
        assert ints(positions_of(placed(positions[number]), tf)) == [p for where in positions[number] for p in where]


def test_a_short_list_is_kept_as_it_is_and_a_long_one_packed() -> None:
    a_long_one = 1000
    columns = of_bodies(["often", "twice"] if n < 2 else ["often"] for n in range(a_long_one))
    made = segment(np.arange(0, 3 * a_long_one, 3, dtype=np.int64), columns, 4.0, Shape())
    held = short_lists(made)
    assert held["twice"][0][:1] == RAW
    assert held["often"][0][:1] == PACKED
    assert len(held["often"][0]) < 100


def test_no_lists_decode_to_nothing() -> None:
    places, counts, bounds = blocked_all([])
    assert (ints(places), ints(counts), bounds) == ([], [], [0])


@pytest.mark.parametrize(
    "ids, refusal",
    [
        ([0, 1 << 32], r"2\*\*32 apart: more than a block holds"),
        ([9, 3], "at rising places, from zero on"),
        ([5, 5], "at rising places, from zero on"),
        ([-1, 5], "at rising places, from zero on"),
    ],
)
def test_chunks_too_far_apart_for_a_block_or_out_of_order_are_refused(ids: list[int], refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        segment(np.array(ids, np.int64), of_bodies([["w"]] * 2), 4.0)


@given(st.lists(st.integers(1, (1 << 63) - 1), min_size=1, max_size=50))
def test_numbers_come_back_from_their_varints(values: list[int]) -> None:
    block = RAW + BLOCK_HEAD.pack(7, len(values)) + (1).to_bytes(4, "little") * (len(values) - 1) + varints(values)
    places, counts = blocked(block)
    assert (ints(places), ints(counts)) == ([*range(7, 7 + len(values))], values)


@pytest.mark.parametrize("value, size", [(0, 1), (127, 1), (128, 2), ((1 << 14) - 1, 2), (1 << 14, 3)])
def test_a_varint_takes_a_byte_per_seven_bits(value: int, size: int) -> None:
    title, body = (["w"], []) if value == 0 else ([], [*["filler"] * (value - 1), "w"])
    made = segment(np.zeros(1, np.int64), [(title, body, [])], 4.0)
    written = short_lists(made)["w"][1]
    assert written == RAW + varint(value)
    assert len(written) == 1 + size


@given(st.lists(st.integers(1, 128), min_size=1, max_size=100), st.integers(0, 1 << 40), st.data())
def test_ranges_hold_where_their_postings_start_and_end_and_their_best_scores(
    counts: list[int], first: int, data: st.DataObject
) -> None:
    found = ascending(data.draw(st.lists(st.integers(1, 1000), min_size=len(counts), max_size=len(counts))), first)
    best = data.draw(st.lists(st.integers(0, 255), min_size=len(counts), max_size=len(counts)))
    back = ranges_of(ranges(ints(found), counts, best))
    assert ints(back.ids) == ints(found)
    assert ints(back.ends - back.starts) == counts
    assert ints(back.starts) == [0, *ints(back.ends[:-1])]
    assert ints(back.best) == best


@given(st.lists(st.integers(0, 5000), min_size=1, max_size=300, unique=True), st.data())
def test_lengths_come_back_by_id_in_blocks_of_consecutive_ids(chosen: list[int], data: st.DataObject) -> None:
    found = np.array(sorted(chosen), np.int64)
    drawn = data.draw(st.lists(st.integers(1, 1 << 20), min_size=len(found), max_size=len(found)))
    lengths = np.array(drawn, np.int64)
    written = with_lengths(found, lengths, shift=6)
    assert set(written) == set(ints(found >> 6))
    for number, held in written.items():
        dense = lengths_of(held, 6)
        inside = np.flatnonzero(np.isin(found >> 6, [number]))
        assert ints(dense[found[inside] & 63]) == ints(lengths[inside])
        assert int(np.count_nonzero(dense)) == len(inside)


def _lengths(data: bytes) -> NDArray[np.int64]:
    return lengths_of(data, 6)


def _decoded_or_refused(data: bytes) -> None:
    with contextlib.suppress(DamagedError):
        places, counts = blocked(data)
        assert len(places) == len(counts)
        assert min(ints(counts), default=1) >= 1
    for decode in (ranges_of, _lengths):
        with contextlib.suppress(DamagedError):
            decode(data)


@given(st.binary(max_size=300))
def test_bytes_that_hold_no_list_are_refused_as_damaged(data: bytes) -> None:
    _decoded_or_refused(data)
    _decoded_or_refused(RAW + data)
    _decoded_or_refused(PACKED + zlib.compress(data))
    _decoded_or_refused(zlib.compress(data))


@given(lists, st.data())
def test_a_list_with_a_byte_changed_or_cut_off_decodes_to_its_own_shape_or_is_refused(
    each: list[Listed], data: st.DataObject
) -> None:
    ids = ints(np.cumsum(np.array([gap for held in each for gap, _ in held], np.int64)))
    counts = [len(where) for held in each for _, where in held]
    raw = block(ids, counts)
    positions = placed([[0] * count for count in counts])
    body = bytearray(zlib.decompress(raw[1:]) if raw[:1] == PACKED else raw[1:])
    body[data.draw(st.integers(0, len(body) - 1))] = data.draw(st.integers(0, 255))
    cut = data.draw(st.integers(0, len(body)))
    for changed in (RAW + bytes(body), RAW + bytes(body[:cut])):
        _decoded_or_refused(changed)
        with contextlib.suppress(DamagedError):
            positions_of(positions, blocked(changed)[1])


@pytest.mark.parametrize(
    "data, refusal",
    [
        (b"", "unknown flag byte"),
        (b"\x02abc", "unknown flag byte"),
        (PACKED + b"not zlib", "does not unpack"),
        (RAW + bytes(5), "shorter than its head"),
        (RAW + BLOCK_HEAD.pack(1, 3), "fewer ids than its head says"),
        (RAW + BLOCK_HEAD.pack(1, 0), "fewer ids than its head says"),
        (RAW + BLOCK_HEAD.pack(1, 1) + b"\x80", "runs past the end"),
        (RAW + BLOCK_HEAD.pack(1, 1) + b"\x81" * 9 + b"\x01", "more than nine bytes"),
        (RAW + BLOCK_HEAD.pack(1, 1) + b"\x00", "a count below one"),
        (RAW + BLOCK_HEAD.pack(1, 1) + b"\x01\x01", "counts other than its ids"),
        (RAW + BLOCK_HEAD.pack(1, 2) + bytes(4) + b"\x01\x01", "ids out of order"),
    ],
)
def test_a_damaged_block_is_refused_with_what_is_wrong(data: bytes, refusal: str) -> None:
    with pytest.raises(DamagedError, match=refusal):
        blocked(data)


def test_damaged_ranges_lengths_and_positions_are_refused_with_what_is_wrong() -> None:
    with pytest.raises(DamagedError, match="does not unpack"):
        ranges_of(b"junk")
    with pytest.raises(DamagedError, match="other than their head says"):
        ranges_of(zlib.compress(RANGES_HEAD.pack(0, 2) + bytes(1)))
    with pytest.raises(DamagedError, match="other than their head says"):
        ranges_of(zlib.compress(bytes(1)))
    with pytest.raises(DamagedError, match="ranges hold ids out of order"):
        ranges_of(zlib.compress(RANGES_HEAD.pack(0, 2) + bytes(4) + bytes(4)))
    with pytest.raises(DamagedError, match="unpacks to other than 256 bytes"):
        lengths_of(zlib.compress(bytes(7)), 6)
    with pytest.raises(DamagedError, match="unpacks to other than 256 bytes"):
        lengths_of(zlib.compress(bytes(1 << 16)), 6)
    with pytest.raises(DamagedError, match="unpacks to other than 256 bytes"):
        lengths_of(zlib.compress(bytes(256))[:-4], 6)
    with pytest.raises(DamagedError, match="positions other than its counts"):
        positions_of(RAW + b"\x01\x02", np.array([3], np.int64))


@pytest.mark.parametrize("gaps", [[2**62], [2**32], [2**31, 2**31], [1, 2**32 - 1]])
def test_positions_beyond_any_passage_are_refused_as_damaged(gaps: list[int]) -> None:
    with pytest.raises(DamagedError, match="a list holds a position past any passage"):
        positions_of(RAW + varints(gaps), np.array([len(gaps)], np.int64))


def test_positions_up_to_the_last_a_passage_holds_are_read() -> None:
    assert ints(positions_of(RAW + varint(2**32 - 1), np.array([1], np.int64))) == [2**32 - 1]


def test_lengths_packed_far_larger_than_their_block_are_never_unpacked_whole() -> None:
    packed = zlib.compress(bytes(1 << 26))
    tracemalloc.start()
    try:
        with pytest.raises(DamagedError, match="unpacks to other than"):
            lengths_of(packed, 6)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1 << 20


def test_numbers_are_kept_little_endian_on_every_machine() -> None:
    (written,) = with_lengths(np.array([0], np.int64), np.array([258], np.int64), 1).values()
    assert zlib.decompress(written) == b"\x02\x01\x00\x00\x00\x00\x00\x00"
    made = segment(np.array([20, 21, 28], np.int64), of_bodies([["w"]] * 3), 4.0, SMALL)
    ((_, _, _, held),) = made.filed.ranged
    assert zlib.decompress(held)[RANGES_HEAD.size : RANGES_HEAD.size + 4] == b"\x02\x00\x00\x00"


def test_a_passage_longer_than_a_length_keeps_is_refused() -> None:
    with pytest.raises(ValueError, match="longer than its length keeps"):
        with_lengths(np.array([0], np.int64), np.array([1 << 32], np.int64), 3)


def test_a_list_that_unpacks_to_more_than_any_list_holds_is_refused_once_it_has_unpacked_that_far() -> None:
    packing = zlib.compressobj()
    piece = bytes(1 << 20)
    packed = b"".join(packing.compress(piece) for _ in range((MOST_INFLATED_BYTES >> 20) + 1)) + packing.flush()
    with pytest.raises(DamagedError, match="unpacks to more than"):
        ranges_of(packed)
    with pytest.raises(DamagedError, match="unpacks to more than"):
        blocked(PACKED + packed)


def test_ranges_too_far_apart_are_refused() -> None:
    with pytest.raises(ValueError, match=r"2\*\*32 apart: more than ranges hold"):
        segment(np.array([0, 1, 1 << 34], np.int64), of_bodies([["w"]] * 3), 4.0, SMALL)
