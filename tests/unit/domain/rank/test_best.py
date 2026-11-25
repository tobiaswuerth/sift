import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.rank.best import Best, may_reach, offered, ranked, unreached_alone

scored = st.lists(
    st.tuples(st.integers(0, 60), st.sampled_from([0.5, 1.0, 1.5, 2.0, 2.5])), max_size=60, unique_by=lambda p: p[0]
)


def batches(pairs: list[tuple[int, float]], size: int) -> list[list[tuple[int, float]]]:
    return [pairs[at : at + size] for at in range(0, len(pairs), size)]


offers = st.builds(batches, scored, st.integers(1, 9))


def brute(each: list[list[tuple[int, float]]], k: int, *, by_document: bool) -> list[tuple[int, float]]:
    found = sorted({pair for offer in each for pair in offer}, key=lambda pair: (-pair[1], pair[0]))
    if by_document:
        seen: set[int] = set()
        found = [pair for pair in found if pair[0] // 4 not in seen and not seen.add(pair[0] // 4)]
    return found[:k]


@given(each=offers, k=st.integers(1, 12), by_document=st.booleans())
def test_the_best_k_are_kept_as_offers_come_ties_by_chunk(
    *, each: list[list[tuple[int, float]]], k: int, by_document: bool
) -> None:
    best = Best(k, by_document)
    for offer in each:
        chunks = np.array([chunk for chunk, _ in offer], np.int64)
        best = offered(best, chunks, chunks // 4, np.array([score for _, score in offer]))
    assert list(ranked(best)) == brute(each, k, by_document=by_document)


def test_a_bound_at_the_threshold_or_below_it_by_the_order_of_a_sum_alone_may_reach_it() -> None:
    threshold = (0.1 + 0.2) + 0.3
    bounds = np.array([threshold, 0.1 + (0.2 + 0.3), threshold * (1 - 1e-6)])
    assert bounds[1] < threshold
    assert may_reach(bounds, threshold).tolist() == [True, True, False]


def test_k_is_one_at_least() -> None:
    with pytest.raises(ValueError, match="k is one at least"):
        Best(0)


def test_the_score_to_beat_is_the_kth_once_there_are_k() -> None:
    best = offered(Best(2, by_document=False), np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([3.0, 1.0, 2.0]))
    assert best.threshold == 2.0
    assert Best(2).threshold == 0.0
    assert offered(best, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)) is best


@given(st.lists(st.floats(0, 100), max_size=8), st.floats(0, 300))
def test_lazy_leaves_alone_never_reach_the_threshold(maxima: list[float], threshold: float) -> None:
    ascending = sorted(maxima)
    lazy = unreached_alone(np.array(ascending), threshold)
    alone = bool(may_reach(np.array([sum(ascending[:lazy])]), threshold).any())
    one_more = lazy == len(ascending) or bool(may_reach(np.array([sum(ascending[: lazy + 1])]), threshold).any())
    assert (lazy > 0 and alone, one_more) == (False, True)


@pytest.mark.parametrize(
    "maxima, threshold, lazy",
    [([], 5.0, 0), ([1.0, 2.0], 0.0, 0), ([1.0, 2.0, 4.0], 3.5, 2), ([1.0, 2.0], 10.0, 2), ([3.0], 3.0, 0)],
)
def test_lazy_leaves_are_the_smallest_whose_sum_stays_below_the_threshold(
    maxima: list[float], threshold: float, lazy: int
) -> None:
    assert unreached_alone(np.array(maxima), threshold) == lazy
