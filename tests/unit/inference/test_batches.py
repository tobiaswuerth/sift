import sys
from collections.abc import Sequence

import numpy as np
import pytest

from sift.domain.codec.codes import Floats
from sift.inference.batches import Attempt, BatchShape, in_batches


class TooLargeError(Exception):
    pass


def lengths(texts: Sequence[str]) -> Floats:
    return np.array([[len(text), 1.0] for text in texts], np.float32)


def too_large(error: BaseException) -> bool:
    return isinstance(error, TooLargeError)


def test_texts_are_run_longest_first_by_their_lengths_in_batches_and_come_back_in_their_order() -> None:
    runs: list[list[str]] = []

    def run(texts: list[str]) -> Floats:
        runs.append(texts)
        return lengths(texts)

    texts = ["bb", "a", "dddd", "ccc", "e"]
    found = in_batches(texts, [len(text) for text in texts], BatchShape(2, 2), Attempt(run, too_large, lambda: None))
    assert runs == [["dddd", "ccc"], ["bb", "a"], ["e"]]
    assert found.rows.dtype == np.float32
    assert found.rows[:, 0].tolist() == [2, 1, 4, 3, 1]
    assert found.batch == 2


def test_a_batch_too_large_for_memory_is_halved_until_it_fits_freeing_memory_each_time() -> None:
    sizes: list[int] = []
    freed: list[BaseException | None] = []

    def run(texts: list[str]) -> Floats:
        sizes.append(len(texts))
        if len(texts) > 2:
            raise TooLargeError
        return lengths(texts)

    found = in_batches(
        ["a"] * 5, [1] * 5, BatchShape(2, 8), Attempt(run, too_large, lambda: freed.append(sys.exc_info()[1]))
    )
    assert sizes == [5, 2, 2, 1]
    assert (found.rows.shape, found.batch) == ((5, 2), 2)
    assert freed == [None]


def test_other_failures_and_a_single_text_too_large_are_raised() -> None:
    def failing(_: list[str]) -> Floats:
        raise ValueError("broken model")

    with pytest.raises(ValueError, match="broken model"):
        in_batches(["a", "b"], [1, 1], BatchShape(2, 2), Attempt(failing, too_large, lambda: None))

    def always(_: list[str]) -> Floats:
        raise TooLargeError

    with pytest.raises(TooLargeError):
        in_batches(["a", "b"], [1, 1], BatchShape(2, 2), Attempt(always, too_large, lambda: None))


def test_no_texts_are_no_rows() -> None:
    assert in_batches([], [], BatchShape(3, 4), Attempt(lengths, too_large, lambda: None)).rows.shape == (0, 3)


def test_a_part_smaller_than_the_batch_is_halved_from_its_own_size() -> None:
    sizes: list[int] = []

    def run(texts: list[str]) -> Floats:
        sizes.append(len(texts))
        if len(texts) > 1:
            raise TooLargeError
        return lengths(texts)

    found = in_batches(["a", "b"], [1, 1], BatchShape(2, 64), Attempt(run, too_large, lambda: None))
    assert (sizes, found.batch) == ([2, 1, 1], 1)
