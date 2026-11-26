import threading
import tracemalloc
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from sift.api.ids import ChunkId
from sift.domain.arrays import Ids, floats, ints
from sift.domain.codec.codes import Bits, Floats, Halves, codes
from sift.retrieval.vector.nearest import (
    SCAN_SHARE,
    SCANNING,
    SHORTLIST,
    Coded,
    ExactRead,
    Near,
    Reads,
    ScanPart,
    coded,
    nearest,
)
from tests.allowed import Chosen
from tests.examples import few

RECALL_AT_10 = 0.75
BYTES_A_CODE = 16


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
    return [Near(ChunkId(chunk), score) for chunk, score in zip(ints(ids[order]), floats(scores[order]), strict=True)]


def reader(exact: list[tuple[Ids, Halves]]) -> ExactRead:
    def read(store: int, chunks: Ids) -> tuple[Ids, Halves]:
        ids, rows = exact[store]
        at = np.searchsorted(ids, chunks)
        return ids[at], rows[at]

    return read


@few(20)
@given(st.integers(0, 2**32 - 1), st.lists(st.integers(0, 60), min_size=1, max_size=3), st.sampled_from([1, 5, 40]))
def test_with_every_candidate_scored_exactly_the_nearest_are_found(seed: int, sizes: list[int], k: int) -> None:
    held, exact = exact_stores(seed, sizes, 24)
    query = exact_query(seed + 1, 24)
    found = nearest(held, query, k, Reads(reader(exact)), rescore=1000)
    assert found == tuple(brute(exact, query, k)) if sum(sizes) else found == ()


@pytest.mark.parametrize("rescore", [1, 3, 1000])
def test_vectors_at_one_distance_are_found_smallest_id_first_however_few_are_rescored(rescore: int) -> None:
    row = unit(np.arange(1, 25, dtype=np.float64))
    rows = np.tile(row, (100, 1)).astype(np.float16)
    exact = [(first + np.arange(0, 300, 3, dtype=np.int64), rows) for first in (10_000, 0)]
    held = [coded(ids, codes(halves.astype(np.float32)), 24) for ids, halves in exact]
    found = nearest(held, row, 7, Reads(reader(exact)), rescore=rescore)
    assert [near.chunk for near in found] == list(range(0, 21, 3))


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
                        {near.chunk for near in nearest(held, query, 10, Reads(reader(exact)), rescore=rescore)}
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


def test_the_codes_find_the_nearest_among_many_more_codes_than_are_scored_exactly() -> None:
    held, exact = stores(9, [200_000], 64)
    rng = np.random.default_rng(10)
    rows = exact[0][1].astype(np.float64)
    picked = ints(rng.integers(200_000, size=20))
    queries = [unit(rows[at : at + 1].reshape(-1) + 0.04 * rng.normal(size=64)) for at in picked]
    found = [{near.chunk for near in nearest(held, query, 10, Reads(reader(exact)), rescore=1000)} for query in queries]
    recall = np.mean(
        [len(each & {near.chunk for near in brute(exact, q, 10)}) / 10 for each, q in zip(found, queries, strict=True)]
    )
    assert recall >= RECALL_AT_10


@pytest.mark.parametrize("sizes", [[200_000], [100_000, 100_000], [30_000] * 6 + [20_000]])
def test_a_scan_without_a_filter_holds_a_few_bytes_a_code_beside_the_codes_however_many_stores(
    sizes: list[int],
) -> None:
    held, exact = stores(11, sizes, 64)
    query = unit(np.random.default_rng(12).normal(size=64))
    nearest(held, query, 10, Reads(reader(exact)), rescore=50)
    tracemalloc.start()
    try:
        nearest(held, query, 10, Reads(reader(exact)), rescore=50)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < BYTES_A_CODE * 200_000


def test_a_scan_of_many_codes_is_spread_over_threads_part_by_part_and_finds_what_one_thread_finds() -> None:
    sizes = [150_000, 40_000, 10, 0]
    held, exact = stores(21, sizes, 64)
    query = unit(np.random.default_rng(22).normal(size=64))
    asked: list[ScanPart] = []

    def one_after_another(scan: Callable[[ScanPart], None], parts: Sequence[ScanPart]) -> None:
        asked.extend(parts)
        for part in parts:
            scan(part)

    alone = nearest(held, query, 10, Reads(reader(exact), spread=one_after_another), rescore=50)
    assert nearest(held, query, 10, Reads(reader(exact)), rescore=50) == alone
    assert len(alone) == 10
    assert [sum(len(part.rows) for part in asked if part.store == store) for store in range(4)] == sizes
    assert sorted(part.at for part in asked) == sorted(sum(sizes[: part.store]) + part.rows.start for part in asked)
    assert (len([part for part in asked if part.store == 0]) > 1, len(asked) <= SCANNING + len(sizes)) == (True, True)


def test_a_scan_of_few_codes_starts_no_thread() -> None:
    held, exact = stores(23, [500, 300], 64)
    query = unit(np.random.default_rng(24).normal(size=64))
    before = threading.active_count()
    seen: list[int] = []

    def counting(store: int, chunks: Ids) -> tuple[Ids, Halves]:
        seen.append(threading.active_count())
        return reader(exact)(store, chunks)

    assert nearest(held, query, 5, Reads(counting), rescore=800) == tuple(brute(exact, query, 5))
    assert seen == [before, before]


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


def exact_query(seed: int, width: int) -> Floats:
    return (np.random.default_rng(seed).integers(-64, 65, size=width) / 64).astype(np.float32)


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
    return [
        Near(ChunkId(chunk), score) for chunk, score in zip(ints(ids[order]), floats(rows[order] @ query), strict=True)
    ]


@few(40)
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
    query = exact_query(seed + 1, width)
    assert list(nearest(held, query, k, Reads(reader(exact)), rescore=rescore)) == staged(exact, query, k, rescore)


def test_a_store_without_codes_is_passed_over_and_not_read() -> None:
    held, exact = exact_stores(3, [80], 24)
    empty = coded(np.zeros(0, np.int64), np.zeros((0, 0), np.uint8), 0)
    read: list[int] = []

    def recorded(store: int, chunks: Ids) -> tuple[Ids, Halves]:
        read.append(store)
        return reader([(np.zeros(0, np.int64), np.zeros((0, 0), np.float16)), *exact])(store, chunks)

    query = exact_query(4, 24)
    assert list(nearest([empty, *held], query, 5, Reads(recorded), rescore=3)) == staged(exact, query, 5, 3)
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
        nearest(held, query, k, Reads(reader(exact)))


def test_codes_alike_are_shortlisted_by_id() -> None:
    rng = np.random.default_rng(6)
    halves = (np.abs(rng.integers(1, 9, size=(50, 24))) / 8).astype(np.float16)
    ids = np.arange(50, dtype=np.int64) * 2
    exact = [(ids, halves)]
    query = (rng.integers(1, 65, size=24) / 64).astype(np.float32)
    found = nearest([coded(ids, codes(halves.astype(np.float32)), 24)], query, 3, Reads(reader(exact)), rescore=1)
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

    assert nearest(held, np.ones(24, np.float32), 0, Reads(refused)) == ()


@pytest.mark.parametrize("share", [0.5, 0.004])
def test_only_codes_allowed_are_found_checking_the_nearest_or_listing_all_for_a_filter_letting_few_through(
    share: float,
) -> None:
    held, exact = stores(11, [30_000, 20_000], 256)
    rng = np.random.default_rng(12)
    masks = [np.isin(ids, rng.choice(ids, max(1, int(share * len(ids))), replace=False)) for ids, _ in exact]
    chosen = [Chosen(ids[mask]) for (ids, _), mask in zip(exact, masks, strict=True)]
    query = unit(rng.normal(size=256))
    found = nearest(held, query, 10, Reads(reader(exact), chosen), rescore=200)
    kept = [(ids[mask], rows[mask]) for (ids, rows), mask in zip(exact, masks, strict=True)]
    alone = [coded(each.ids[mask], each.bits[mask], 256) for each, mask in zip(held, masks, strict=True)]
    assert found == nearest(alone, query, 10, Reads(reader(kept)), rescore=200)
    if share > SCAN_SHARE:
        assert not any(each.listings for each in chosen)
        assert sum(sum(each.asked) for each in chosen) <= SCAN_SHARE * 50_000
    else:
        assert all(each.listings == [True] for each in chosen)


def test_a_store_without_a_filter_beside_one_with_it_keeps_every_code() -> None:
    held, exact = stores(13, [3000, 3000], 256)
    rng = np.random.default_rng(14)
    mask = np.isin(exact[1][0], rng.choice(exact[1][0], 1500, replace=False))
    chosen = Chosen(exact[1][0][mask])
    query = unit(rng.normal(size=256))
    found = nearest(held, query, 10, Reads(reader(exact), [None, chosen]), rescore=10)
    kept = [exact[0], (exact[1][0][mask], exact[1][1][mask])]
    alone = [held[0], coded(held[1].ids[mask], held[1].bits[mask], 256)]
    assert found == nearest(alone, query, 10, Reads(reader(kept)), rescore=10)
    assert (chosen.listings, len(chosen.asked) > 0) == ([], True)
    few = Chosen(exact[1][0][:5])
    listed = nearest(held, query, 3005, Reads(reader(exact), [None, few]), rescore=3005)
    assert {near.chunk for near in listed} == {*ints(exact[0][0]), *ints(exact[1][0][:5])}
    assert few.listings == [True]


def test_a_filter_over_a_store_without_codes_is_never_listed_and_one_letting_nothing_through_finds_nothing() -> None:
    held, exact = stores(15, [3000, 0], 256)
    query = unit(np.random.default_rng(16).normal(size=256))
    few, empty = Chosen(exact[0][0][:5]), Chosen(np.zeros(0, np.int64))
    found = nearest(held, query, 3005, Reads(reader(exact), [few, empty]), rescore=3005)
    assert {near.chunk for near in found} == set(ints(exact[0][0][:5]))
    assert (few.listings, empty.listings) == ([True], [])
    nothing = Chosen(np.zeros(0, np.int64))
    assert nearest(held, query, 10, Reads(reader(exact), [nothing, None]), rescore=10) == ()
