import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.arrays import among, distinct, ints

numbers = st.lists(st.integers(-(1 << 62), 1 << 62), max_size=60)


@given(numbers, numbers)
def test_among_marks_the_values_a_sorted_set_holds(values: list[int], held: list[int]) -> None:
    ascending = np.unique(np.array(held, np.int64))
    found = among(np.array(values, np.int64), ascending)
    assert ints(np.flatnonzero(found)) == [at for at, value in enumerate(values) if value in set(held)]


@given(numbers)
def test_distinct_keeps_each_value_of_an_ascending_array_once(values: list[int]) -> None:
    ascending = np.sort(np.array(values, np.int64))
    assert ints(distinct(ascending)) == sorted(set(values))
