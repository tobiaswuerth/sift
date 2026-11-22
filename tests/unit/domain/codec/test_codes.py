import tracemalloc
from typing import cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from sift.domain.arrays import ints
from sift.domain.codec.codes import Floats, bit_scores, code_of, codes, distances, width_of, words

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
    found = distances(words(codes(matrix)), code_of(query))
    assert found.dtype == np.uint16
    assert ints(found) == ints(np.not_equal(matrix > 0, query > 0).sum(axis=1))


@given(matrices, st.data())
def test_bits_are_scored_as_the_query_against_their_signs(matrix: Floats, data: st.DataObject) -> None:
    query = cast("Floats", data.draw(hnp.arrays(np.float32, width_of(matrix), elements=st.floats(-1, 1, width=32))))
    signs = np.where(matrix > 0, 1.0, -1.0)
    assert bit_scores(codes(matrix), query) == pytest.approx(signs @ query, abs=1e-4)


def test_codes_of_no_rows_are_none() -> None:
    assert codes(np.zeros((0, 8), np.float32)).shape == (0, 8)
    assert distances(np.zeros((0, 1), np.uint64), np.zeros(1, np.uint64)).shape == (0,)


def test_distances_are_scanned_in_blocks_without_a_copy_of_the_codes() -> None:
    rng = np.random.default_rng(5)
    held = words(rng.integers(0, 256, size=(400_000, 16), dtype=np.uint8))
    query = words(rng.integers(0, 256, size=(1, 16), dtype=np.uint8)).reshape(-1)
    tracemalloc.start()
    found = distances(held, query, scanned=1 << 20)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert ints(found) == ints(np.bitwise_count(held ^ query).sum(axis=1))
    assert peak < held.nbytes / 1.5
