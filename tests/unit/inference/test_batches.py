import sys
import threading
from collections.abc import Sequence

import numpy as np
import pytest

from sift.domain.codec.codes import Floats
from sift.inference.batches import Attempt, BatchShape, in_batches
from tests.waits import paused, waited


class TooLargeError(Exception):
    pass


def lengths(texts: Sequence[str]) -> Floats:
    return np.array([[len(text), 1.0] for text in texts], np.float32)


def too_large(error: BaseException) -> bool:
    return isinstance(error, TooLargeError)


def prepared(texts: list[str]) -> list[str]:
    return texts


def test_texts_are_run_longest_first_by_their_lengths_in_batches_and_come_back_in_their_order() -> None:
    runs: list[list[str]] = []

    def run(texts: list[str]) -> Floats:
        runs.append(texts)
        return lengths(texts)

    texts = ["bb", "a", "dddd", "ccc", "e"]
    found = in_batches(
        texts, [len(text) for text in texts], BatchShape(2, 2), Attempt(prepared, run, too_large, lambda: None)
    )
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
        ["a"] * 5, [1] * 5, BatchShape(2, 8), Attempt(prepared, run, too_large, lambda: freed.append(sys.exc_info()[1]))
    )
    assert sizes == [5, 2, 2, 1]
    assert (found.rows.shape, found.batch) == ((5, 2), 2)
    assert freed == [None]


def test_other_failures_and_a_single_text_too_large_are_raised() -> None:
    def failing(_: list[str]) -> Floats:
        raise ValueError("broken model")

    with pytest.raises(ValueError, match="broken model"):
        in_batches(["a", "b"], [1, 1], BatchShape(2, 2), Attempt(prepared, failing, too_large, lambda: None))

    def always(_: list[str]) -> Floats:
        raise TooLargeError

    with pytest.raises(TooLargeError):
        in_batches(["a", "b"], [1, 1], BatchShape(2, 2), Attempt(prepared, always, too_large, lambda: None))


def test_recursion_too_deep_is_raised_at_once_whatever_it_says() -> None:
    def too_deep(_: list[str]) -> Floats:
        raise RecursionError("out of memory")

    freed: list[None] = []
    with pytest.raises(RecursionError):
        in_batches(
            ["a", "b"],
            [1, 1],
            BatchShape(2, 2),
            Attempt(prepared, too_deep, lambda _: True, lambda: freed.append(None)),
        )
    assert freed == []


def test_no_texts_are_no_rows() -> None:
    assert in_batches([], [], BatchShape(3, 4), Attempt(prepared, lengths, too_large, lambda: None)).rows.shape == (
        0,
        3,
    )


def test_a_part_smaller_than_the_batch_is_halved_from_its_own_size() -> None:
    sizes: list[int] = []

    def run(texts: list[str]) -> Floats:
        sizes.append(len(texts))
        if len(texts) > 1:
            raise TooLargeError
        return lengths(texts)

    found = in_batches(["a", "b"], [1, 1], BatchShape(2, 64), Attempt(prepared, run, too_large, lambda: None))
    assert (sizes, found.batch) == ([2, 1, 1], 1)


def test_the_next_batches_are_prepared_on_another_thread_while_one_runs() -> None:
    second_ready, threads = threading.Event(), set[int]()

    def preparing(texts: list[str]) -> list[str]:
        threads.add(threading.get_ident())
        if texts == ["b"]:
            second_ready.set()
        return texts

    def run(texts: list[str]) -> Floats:
        if texts == ["a"]:
            waited(second_ready)
        return lengths(texts)

    found = in_batches(["a", "b", "c"], [3, 2, 1], BatchShape(2, 1), Attempt(preparing, run, too_large, lambda: None))
    assert (found.rows[:, 0].tolist(), threading.get_ident() in threads, len(threads)) == ([1, 1, 1], False, 1)


def test_a_batch_that_cannot_be_prepared_raises_its_error_where_the_batches_run() -> None:
    def preparing(texts: list[str]) -> list[str]:
        if texts == ["b"]:
            raise ValueError("no tokens for b")
        return texts

    ran: list[list[str]] = []

    def run(texts: list[str]) -> Floats:
        ran.append(texts)
        return lengths(texts)

    with pytest.raises(ValueError, match="no tokens for b"):
        in_batches(["a", "b", "c"], [3, 2, 1], BatchShape(2, 1), Attempt(preparing, run, too_large, lambda: None))
    assert ran == [["a"]]


def test_a_batch_halved_after_running_out_of_memory_is_prepared_anew_in_its_halves() -> None:
    prepared_parts: list[list[str]] = []

    def preparing(texts: list[str]) -> list[str]:
        prepared_parts.append(texts)
        return texts

    def run(texts: list[str]) -> Floats:
        if len(texts) > 2:
            raise TooLargeError
        return lengths(texts)

    texts = ["dddd", "ccc", "bb", "a"]
    found = in_batches(
        texts, [len(text) for text in texts], BatchShape(2, 4), Attempt(preparing, run, too_large, lambda: None)
    )
    assert prepared_parts == [texts, ["dddd", "ccc"], ["bb", "a"]]
    assert (found.rows[:, 0].tolist(), found.batch) == ([4, 3, 2, 1], 2)


def test_a_recursion_too_deep_while_a_batch_is_prepared_is_raised_where_the_batches_run() -> None:
    def preparing(texts: list[str]) -> list[str]:
        if texts == ["b"]:
            raise RecursionError("maximum recursion depth exceeded")
        return texts

    with pytest.raises(RecursionError):
        in_batches(["a", "b"], [2, 1], BatchShape(2, 1), Attempt(preparing, lengths, too_large, lambda: None))


def test_a_batch_not_prepared_in_time_says_the_tokenizer_is_stuck() -> None:
    def preparing(texts: list[str]) -> list[str]:
        paused(0.5)
        return texts

    with pytest.raises(TimeoutError, match="a batch was not ready within 0 s: the tokenizer is stuck"):
        in_batches(["a"], [1], BatchShape(2, 1), Attempt(preparing, lengths, too_large, lambda: None), waited=0.01)
