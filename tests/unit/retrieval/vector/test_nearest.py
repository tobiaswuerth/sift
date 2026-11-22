import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from sift.domain.arrays import Ids, floats, ints
from sift.domain.codec.codes import Floats, codes
from sift.retrieval.vector.nearest import Coded, Exact, Near, coded, nearest
from sift.storage.vectors import Halves


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
        held.append(coded(ids, codes(halves.astype(np.float32))))
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
