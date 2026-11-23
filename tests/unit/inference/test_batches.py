from collections.abc import Sequence

import numpy as np
import pytest

from sift.domain.codec.codes import Floats
from sift.inference.batches import Shape, in_batches


class TooLargeError(Exception):
    pass


def lengths(texts: Sequence[str]) -> Floats:
    return np.array([[len(text), 1.0] for text in texts], np.float32)


def test_texts_are_run_longest_first_by_their_lengths_in_batches_and_come_back_in_their_order() -> None:
    runs: list[list[str]] = []

    def run(texts: list[str]) -> Floats:
        runs.append(texts)
        return lengths(texts)

    texts = ["bb", "a", "dddd", "ccc", "e"]
    rows = in_batches(texts, [len(text) for text in texts], Shape(2, 2), run, lambda _: False)
    assert runs == [["dddd", "ccc"], ["bb", "a"], ["e"]]
    assert rows.dtype == np.float16
    assert rows[:, 0].tolist() == [2, 1, 4, 3, 1]


def test_a_batch_too_large_for_memory_is_halved_until_it_fits() -> None:
    sizes: list[int] = []

    def run(texts: list[str]) -> Floats:
        sizes.append(len(texts))
        if len(texts) > 2:
            raise TooLargeError
        return lengths(texts)

    rows = in_batches(["a"] * 5, [1] * 5, Shape(2, 8), run, lambda error: isinstance(error, TooLargeError))
    assert sizes == [5, 4, 2, 2, 1]
    assert rows.shape == (5, 2)


def test_other_failures_and_a_single_text_too_large_are_raised() -> None:
    def failing(_: list[str]) -> Floats:
        raise ValueError("broken model")

    with pytest.raises(ValueError, match="broken model"):
        in_batches(["a", "b"], [1, 1], Shape(2, 2), failing, lambda error: isinstance(error, TooLargeError))

    def too_large(_: list[str]) -> Floats:
        raise TooLargeError

    with pytest.raises(TooLargeError):
        in_batches(["a", "b"], [1, 1], Shape(2, 2), too_large, lambda error: isinstance(error, TooLargeError))


def test_no_texts_are_no_rows() -> None:
    assert in_batches([], [], Shape(3, 4), lengths, lambda _: False).shape == (0, 3)
