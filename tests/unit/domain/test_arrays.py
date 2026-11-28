import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.arrays import Ascending, among, ascending, distinct, ints, read_only
from sift.domain.codec.codes import BYTE_BITS

numbers = st.lists(st.integers(-(1 << 62), 1 << 62), max_size=60)


@given(numbers, numbers)
def test_among_marks_the_values_a_sorted_set_holds(values: list[int], held: list[int]) -> None:
    found = among(np.array(values, np.int64), ascending(np.array(held, np.int64)))
    assert ints(np.flatnonzero(found)) == [at for at, value in enumerate(values) if value in set(held)]


@given(numbers)
def test_distinct_keeps_each_value_of_an_ascending_array_once(values: list[int]) -> None:
    assert ints(distinct(ascending(np.array(values, np.int64))).values) == sorted(set(values))


def test_values_out_of_order_are_never_taken_for_ascending() -> None:
    with pytest.raises(ValueError, match="not ascending"):
        Ascending(np.array([2, 1], np.int64))


def test_an_array_made_read_only_refuses_writes_as_do_the_module_tables() -> None:
    for table in (read_only(np.zeros(2)), BYTE_BITS):
        with pytest.raises(ValueError, match="read-only"):
            table[0] = 1
