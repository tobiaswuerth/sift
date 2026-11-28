import tracemalloc
from typing import cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from sift.domain.arrays import ints
from sift.domain.codec.codes import (
    Floats,
    bit_scores,
    code_of,
    codes,
    filled,
    filling,
    finished,
    measured,
    packed,
    rows_of,
    width_of,
)

matrices = st.integers(1, 12).flatmap(
    lambda rows: st.integers(1, 130).flatmap(
        lambda width: hnp.arrays(np.float32, (rows, width), elements=st.floats(-1, 1, width=32))
    )
)


@given(matrices)
def test_a_code_holds_a_bit_a_dimension_set_where_the_value_is_positive_in_whole_words(matrix: Floats) -> None:
    found = codes(matrix)
    width = width_of(matrix)
    assert found.dtype == np.uint8
    assert found.shape == (len(matrix), -(-width // 64) * 8)
    unpacked = np.unpackbits(found, axis=1)
    assert bool(np.array_equal(unpacked[:, :width], matrix > 0))
    assert not bool(unpacked[:, width:].any())


@given(matrices, st.data())
def test_the_distance_of_two_codes_is_the_number_of_signs_they_differ_in(matrix: Floats, data: st.DataObject) -> None:
    query = cast("Floats", data.draw(hnp.arrays(np.float32, width_of(matrix), elements=st.floats(-1, 1, width=32))))
    found, counted = measured([packed(codes(matrix))], code_of(query))
    assert found.dtype == np.uint16
    assert ints(found) == ints(np.not_equal(matrix > 0, query > 0).sum(axis=1))
    assert ints(counted) == ints(np.bincount(found, minlength=8 * width_of(codes(matrix)) + 1))


@given(matrices, st.data())
def test_bits_are_scored_as_the_query_against_their_signs(matrix: Floats, data: st.DataObject) -> None:
    query = cast("Floats", data.draw(hnp.arrays(np.float32, width_of(matrix), elements=st.floats(-1, 1, width=32))))
    signs = np.where(matrix > 0, 1.0, -1.0)
    assert bit_scores(codes(matrix), query) == pytest.approx(signs @ query, abs=1e-4)


def test_codes_of_no_rows_are_none() -> None:
    empty = packed(codes(np.zeros((0, 8), np.float32)))
    assert (codes(np.zeros((0, 8), np.float32)).shape, (empty.rows, empty.size)) == ((0, 8), (0, 8))
    found, counted = measured([empty], code_of(np.ones(8, np.float32)))
    assert (found.shape, ints(counted)) == ((0,), [0] * 65)


def test_the_codes_of_several_stores_are_scanned_in_one_and_their_rows_gathered_by_number() -> None:
    rng = np.random.default_rng(3)
    first, second = (rng.integers(0, 256, size=(rows, 16), dtype=np.uint8) for rows in (40_000, 5))
    query = rng.integers(0, 256, size=16, dtype=np.uint8)
    held = [packed(first), packed(np.zeros((0, 0), np.uint8)), packed(second)]
    found, counted = measured(held, query.view(np.uint64))
    both = np.concatenate([first, second])
    assert ints(found) == ints(np.bitwise_count(both ^ query).sum(axis=1))
    assert int(counted.sum()) == len(both)
    assert np.array_equal(rows_of(held[0], np.array([7, 0, 7])), first[[7, 0, 7]])
    with pytest.raises(ValueError, match="do not hold"):
        rows_of(held[2], np.array([5]))


def test_codes_are_filled_whole_rows_at_a_time_and_scanned_with_a_query_as_wide() -> None:
    for rows, width in ((1 << 62, 8), (-1, 8)):
        with pytest.raises((ValueError, OverflowError), match=r"more rows than memory|can't convert negative"):
            filling(rows, width)
    codes_of_two = filling(2, 8)
    with pytest.raises(ValueError, match="before all the rows"):
        finished(codes_of_two)
    for wrong in (bytes(4), bytes(24)):
        with pytest.raises(ValueError, match="part of a row"):
            filled(codes_of_two, wrong)
    filled(codes_of_two, bytes(16))
    held = finished(codes_of_two)
    assert (held.rows, held.size) == (2, 8)
    with pytest.raises(ValueError, match="before all the rows"):
        finished(codes_of_two)
    with pytest.raises(ValueError, match="whole words"):
        measured([held], np.zeros(2, np.uint64))


@pytest.mark.parametrize("rows, size", [(400_000, 16), (3_000, 8_192)])
def test_distances_are_scanned_without_a_copy_of_the_codes(rows: int, size: int) -> None:
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 256, size=(rows, size), dtype=np.uint8)
    query = rng.integers(0, 256, size=size, dtype=np.uint8)
    held = packed(bits)
    tracemalloc.start()
    found, _ = measured([held], query.view(np.uint64))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert ints(found) == ints(np.bitwise_count(bits ^ query).sum(axis=1))
    assert peak < bits.nbytes / 1.5
