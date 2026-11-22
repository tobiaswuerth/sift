from typing import cast

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from sift.domain.arrays import Ids, floats, ints
from sift.domain.codec.codes import Bits, Floats, Halves, codes
from sift.retrieval.vector.nearest import SHORTLIST, Coded, Exact, Near, coded, nearest


def unit(values: NDArray[np.float64]) -> Floats:
    return (values / np.linalg.norm(values)).astype(np.float32)


def stores(seed: int, sizes: list[int], width: int) -> tuple[list[Coded], list[tuple[Ids, Halves]]]:
    rng = np.random.default_rng(seed)
    held: list[Coded] = []
    exact: list[tuple[Ids, Halves]] = []
    first = 0
    for size in sizes:
        matrix = rng.normal(size=(size, width)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        ids = first + np.arange(size, dtype=np.int64) * 3
        first += 10_000
        halves = matrix.astype(np.float16)
        held.append(coded(ids, codes(halves.astype(np.float32)), width))
        exact.append((ids, halves))
    return held, exact


def brute(exact: list[tuple[Ids, Halves]], query: Floats, k: int) -> list[Near]:
    ids = np.concatenate([each for each, _ in exact])
    scores = np.concatenate([rows.astype(np.float32) @ query for _, rows in exact])
    order = np.lexsort((ids, -scores))[:k]
    return [Near(chunk, score) for chunk, score in zip(ints(ids[order]), floats(scores[order]), strict=True)]


def reader(exact: list[tuple[Ids, Halves]]) -> Exact:
    def read(store: int, chunks: Ids) -> tuple[Ids, Halves]:
        ids, rows = exact[store]
        at = np.searchsorted(ids, chunks)
        return ids[at], rows[at]

    return read


@settings(max_examples=20, deadline=None)
@given(st.integers(0, 2**32 - 1), st.lists(st.integers(0, 60), min_size=1, max_size=3), st.sampled_from([1, 5, 40]))
def test_with_every_candidate_scored_exactly_the_nearest_are_found(seed: int, sizes: list[int], k: int) -> None:
    held, exact = stores(seed, sizes, 24)
    query = unit(np.random.default_rng(seed + 1).normal(size=24))
    found = nearest(held, query, k, reader(exact), rescore=1000)
    assert found == tuple(brute(exact, query, k)) if sum(sizes) else found == ()


def test_the_shortlist_by_codes_finds_more_of_the_nearest_the_more_it_rescores() -> None:
    held, exact = stores(7, [3000, 2000], 64)
    rng = np.random.default_rng(8)
    queries: list[Floats] = []
    for _ in range(20):
        at = int(rng.integers(3000))
        near = exact[0][1][at : at + 1].astype(np.float64).reshape(-1)
        queries.append(unit(near + 0.3 * rng.normal(size=64)))
    recall = [
        float(
            np.mean(
                [
                    len(
                        {near.chunk for near in nearest(held, query, 10, reader(exact), rescore=rescore)}
                        & {near.chunk for near in brute(exact, query, 10)}
                    )
                    / 10
                    for query in queries
                ]
            )
        )
        for rescore in (10, 100, 300, 1000)
    ]
    assert recall == sorted(recall)
    assert recall[2] >= 0.9
    assert recall[3] == 1.0


def exact_stores(seed: int, sizes: list[int], width: int) -> tuple[list[Coded], list[tuple[Ids, Halves]]]:
    rng = np.random.default_rng(seed)
    held: list[Coded] = []
    exact: list[tuple[Ids, Halves]] = []
    for at, size in enumerate(sizes):
        halves = (rng.integers(-8, 9, size=(size, width)) / 8).astype(np.float16)
        copies = rng.integers(0, max(size, 1), size=size // 4)
        halves[: len(copies)] = halves[copies]
        ids = at * 10_000 + np.arange(size, dtype=np.int64) * 3
        held.append(coded(ids, codes(halves.astype(np.float32)), width))
        exact.append((ids, halves))
    return held, exact


def staged(exact: list[tuple[Ids, Halves]], query: Floats, k: int, rescore: int) -> list[Near]:
    ids = np.concatenate([each for each, _ in exact])
    signs = np.vstack([rows for _, rows in exact]) > 0
    rows = np.vstack([rows for _, rows in exact]).astype(np.float32)
    want = max(rescore, k)
    hamming: Ids = np.sum(np.not_equal(signs, np.greater(query, 0)), axis=1, dtype=np.int64)
    short = np.lexsort((ids, hamming))[: SHORTLIST * want]
    by_bits = np.where(signs[short], query, -query).sum(axis=1)
    best = short[np.lexsort((ids[short], -by_bits))[:want]]
    scores = rows[best] @ query
    order = best[np.lexsort((ids[best], -scores))[:k]]
    return [Near(chunk, score) for chunk, score in zip(ints(ids[order]), floats(rows[order] @ query), strict=True)]


@settings(max_examples=40, deadline=None)
@given(
    st.integers(0, 2**32 - 1),
    st.lists(st.integers(0, 300), min_size=1, max_size=3),
    st.sampled_from([8, 24, 70]),
    st.integers(0, 12),
    st.integers(1, 25),
)
def test_the_nearest_are_those_by_codes_then_by_bits_then_exactly_with_ties_by_id(
    seed: int, sizes: list[int], width: int, k: int, rescore: int
) -> None:
    held, exact = exact_stores(seed, sizes, width)
    query = (np.random.default_rng(seed + 1).integers(-64, 65, size=width) / 64).astype(np.float32)
    assert list(nearest(held, query, k, reader(exact), rescore=rescore)) == staged(exact, query, k, rescore)


def test_a_store_without_codes_is_passed_over_and_not_read() -> None:
    held, exact = stores(3, [80], 24)
    empty = coded(np.zeros(0, np.int64), np.zeros((0, 0), np.uint8), 0)
    read: list[int] = []

    def recorded(store: int, chunks: Ids) -> tuple[Ids, Halves]:
        read.append(store)
        return reader([(np.zeros(0, np.int64), np.zeros((0, 0), np.float16)), *exact])(store, chunks)

    query = unit(np.random.default_rng(4).normal(size=24))
    assert list(nearest([empty, *held], query, 5, recorded, rescore=3)) == staged(exact, query, 5, 3)
    assert read == [1]


@pytest.mark.parametrize(
    "query, k, refusal",
    [
        (np.full(24, np.nan, np.float32), 5, "not one vector of finite numbers"),
        (np.ones((1, 24), np.float32), 5, "not one vector of finite numbers"),
        (np.ones(16, np.float32), 5, "the query has 16 dimensions, unlike the vectors"),
        (np.ones(24, np.float32), -1, "-1 nearest were asked for"),
    ],
)
def test_a_query_that_cannot_be_compared_is_refused(query: Floats, k: int, refusal: str) -> None:
    held, exact = stores(3, [10], 24)
    with pytest.raises(ValueError, match=refusal):
        nearest(held, query, k, reader(exact))


def test_codes_alike_are_shortlisted_by_id() -> None:
    rng = np.random.default_rng(6)
    halves = (np.abs(rng.integers(1, 9, size=(50, 24))) / 8).astype(np.float16)
    ids = np.arange(50, dtype=np.int64) * 2
    exact = [(ids, halves)]
    query = (rng.integers(1, 65, size=24) / 64).astype(np.float32)
    found = nearest([coded(ids, codes(halves.astype(np.float32)), 24)], query, 3, reader(exact), rescore=1)
    assert list(found) == staged(exact, query, 3, 1)
    assert {near.chunk for near in found} <= set(ints(ids[:3]))


def test_codes_not_of_their_width_are_refused() -> None:
    with pytest.raises(ValueError, match="codes of 64 dimensions are 8 bytes each"):
        coded(np.arange(2, dtype=np.int64), codes(np.ones((2, 128), np.float32)), 64)
    with pytest.raises(ValueError, match="a row a chunk"):
        coded(np.arange(3, dtype=np.int64), codes(np.ones((2, 64), np.float32)), 64)
    with pytest.raises(ValueError, match="uint8 matrix"):
        coded(np.arange(2, dtype=np.int64), cast("Bits", codes(np.ones((2, 64), np.float32)).astype(np.int8)), 64)
    with pytest.raises(ValueError, match="codes of 65,536 dimensions"):
        coded(np.arange(1, dtype=np.int64), np.zeros((1, 8_192), np.uint8), 65_536)
    with pytest.raises(ValueError, match="uint8 matrix"):
        coded(np.arange(8, dtype=np.int64), np.zeros(8, np.uint8), 64)
    with pytest.raises(ValueError, match="codes of 0 dimensions"):
        coded(np.arange(2, dtype=np.int64), np.zeros((2, 0), np.uint8), 0)


def test_a_store_without_codes_may_be_of_any_width() -> None:
    assert len(coded(np.zeros(0, np.int64), np.zeros((0, 0), np.uint8), 24).ids) == 0


def test_none_asked_for_reads_nothing() -> None:
    held, _ = stores(3, [10], 24)

    def refused(store: int, chunks: Ids) -> tuple[Ids, Halves]:
        raise AssertionError((store, chunks))

    assert nearest(held, np.ones(24, np.float32), 0, refused) == ()
