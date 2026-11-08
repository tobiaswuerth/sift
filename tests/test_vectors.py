"""The stores' vectors in memory as one bit per dimension, its sign: candidates by Hamming distance, among the codes of
every collection searched, the best of them scored exactly with the vectors the stores hold."""

import numpy as np
import pytest

from sift import vectors
from sift.vectors import Codes, codes


def unit(matrix):
    matrix = np.asarray(matrix, np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


class Exact:
    """The exact vectors the index holds, by chunk id, as fp16; counts how many it was asked for."""

    def __init__(self, ids, matrix):
        self.rows = dict(zip(np.asarray(ids).tolist(), np.asarray(matrix, np.float16)))
        self.asked = []

    def __call__(self, chunks):
        chunks = np.asarray(chunks).tolist()
        self.asked.append(len(chunks))
        found = [c for c in chunks if c in self.rows]
        dim = len(next(iter(self.rows.values()))) if self.rows else 0
        return np.array(found, np.int64), np.array([self.rows[c] for c in found], np.float16).reshape(len(found), dim)


def random_index(n=500, dim=64, seed=1):
    rng = np.random.default_rng(seed)
    matrix = unit(rng.standard_normal((n, dim))).astype(np.float16)
    ids = np.arange(10, 10 * (n + 1), 10, dtype=np.int64)  # sorted, with gaps, as chunk ids are
    exact = Exact(ids, matrix)
    return ids, matrix, exact, Codes(ids, codes(matrix))


def top(held, exact, query, k, **kwargs):
    """The k nearest among the codes `held`, one Codes or a list, one per collection."""
    return vectors.top(held if isinstance(held, list) else [held], query, k, exact, **kwargs)


def exact_top(ids, matrix, query, k, keep=None):
    scores = matrix.astype(np.float32) @ query
    order = [i for i in np.argsort(-scores, kind="stable") if keep is None or ids[i] in keep][:k]
    return [int(ids[i]) for i in order]


def chunks(ranked):
    return [chunk for chunk, _ in ranked]


def test_codes_are_the_signs_one_bit_a_dimension_in_whole_words():
    bits = codes(np.array([[0.5, -0.1, 0, 0.2, -1, 1, 1, -0.3, 0.7]], np.float16))
    assert bits.dtype == np.uint8
    assert bits.shape == (1, 8)  # 9 dimensions take 2 bytes, padded to one 8-byte word
    assert bits[0, :2].tolist() == [0b10010110, 0b10000000]  # positive is 1; zero counts as negative
    assert codes(np.zeros((3, 768), np.float16)).shape == (3, 96)


def test_with_every_vector_a_candidate_top_is_exact_search():
    ids, matrix, exact, index = random_index()
    for query in unit(np.random.default_rng(2).standard_normal((20, 64))):
        ranked = top(index, exact, query, 10)
        assert chunks(ranked) == exact_top(ids, matrix, query, 10)
        expected = (matrix.astype(np.float32) @ query)[np.searchsorted(ids, chunks(ranked))]
        assert [score for _, score in ranked] == pytest.approx(expected.tolist(), abs=1e-3)  # exact cosines


def test_with_few_candidates_the_nearest_still_come_first(monkeypatch):
    # Over 3.2M arXiv vectors, 4,000 candidates found 99.6% of exact search's top 10 (docs/benchmarks.md).
    monkeypatch.setattr(vectors, "RESCORE", 100)
    ids, matrix, exact, index = random_index(n=5000, dim=128)
    rng = np.random.default_rng(3)
    for i in rng.choice(5000, 30, replace=False):  # a query close to one passage finds it first
        query = unit(matrix[i : i + 1].astype(np.float32) + 0.05 * rng.standard_normal((1, 128)))[0]
        assert chunks(top(index, exact, query, 1)) == [ids[i]]
    assert exact.asked == [100] * 30  # only the candidates are read from the index


def test_top_keeps_to_the_allowed_chunks_and_leaves_out_the_excluded():
    ids, matrix, exact, index = random_index()
    query = unit(np.ones((1, 64)))[0]
    allowed = ids[::3]
    ranked = top(index, exact, query, 5, allowed=allowed, excluded=allowed[:2])
    keep = set(allowed.tolist()) - set(allowed[:2].tolist())
    assert chunks(ranked) == exact_top(ids, matrix, query, 5, keep)


def test_allowed_chunks_far_from_the_query_are_still_found(monkeypatch):
    # A filter applies before the candidates are picked: a small collection among millions isn't crowded out.
    monkeypatch.setattr(vectors, "RESCORE", 10)
    ids, matrix, exact, index = random_index(n=2000)
    query = unit(np.ones((1, 64)))[0]
    farthest = [int(ids[i]) for i in np.argsort(matrix.astype(np.float32) @ query)[:3]]
    assert sorted(chunks(top(index, exact, query, 10, allowed=np.array(farthest)))) == sorted(farthest)


def test_the_float_query_picks_the_candidates_scored_exactly(monkeypatch):
    # The query's own floats against the candidates' sign bits rank them better than Hamming distance: over 3.2M arXiv
    # vectors, 1,000 of the 10,000 nearest scored exactly found more of exact search's top 10 than 4,000 by Hamming.
    monkeypatch.setattr(vectors, "RESCORE", 20)
    ids, matrix, exact, index = random_index(n=5000, dim=128)
    queries = unit(np.random.default_rng(9).standard_normal((40, 128)))

    def recall():
        return np.mean(
            [len(set(chunks(top(index, exact, q, 10))) & set(exact_top(ids, matrix, q, 10))) for q in queries]
        )

    monkeypatch.setattr(vectors, "SHORTLIST", 1)  # Hamming distance alone
    by_hamming = recall()
    monkeypatch.setattr(vectors, "SHORTLIST", 10)
    exact.asked.clear()
    assert recall() > by_hamming + 0.5
    assert set(exact.asked) == {20}  # still only these scored exactly


def alike(got, expected):
    """The same chunks with the same cosines, to rounding: other candidates, another product of matrices."""
    assert chunks(got) == chunks(expected)
    assert [score for _, score in got] == pytest.approx([score for _, score in expected], abs=1e-6)


class Allowed(vectors.Filter):
    """A filter letting `ids` through; counts the chunks it checked and how often it listed them all."""

    def __init__(self, ids):
        self.ids, self.checked, self.lists = set(np.asarray(ids).tolist()), 0, 0

    def check(self, ids):
        self.checked += len(ids)
        return np.array([i in self.ids for i in np.asarray(ids).tolist()], bool)

    def listed(self):
        self.lists += 1
        return np.array(sorted(self.ids), np.int64)


def test_a_filter_letting_many_through_checks_the_nearest_candidates_alone(monkeypatch):
    # Listing the chunks a filter lets through reads every document of the store, seconds over millions: checking the
    # nearest few thousand candidates one by one finds the same.
    monkeypatch.setattr(vectors, "RESCORE", 5)  # k of 10: 100 candidates must pass
    ids, _, exact, index = random_index(n=6000)
    for query in unit(np.random.default_rng(7).standard_normal((10, 64))):
        allowed = Allowed(ids[::3])
        alike(top(index, exact, query, 10, allowed=allowed), top(index, exact, query, 10, allowed=ids[::3]))
        assert allowed.lists == 0
        assert allowed.checked < len(ids) / 5


def test_a_filter_letting_few_through_lists_them(monkeypatch):
    monkeypatch.setattr(vectors, "RESCORE", 10)
    ids, matrix, exact, index = random_index(n=2000)
    query = unit(np.ones((1, 64)))[0]
    farthest = [int(ids[i]) for i in np.argsort(matrix.astype(np.float32) @ query)[:3]]
    allowed = Allowed(farthest)
    assert sorted(chunks(top(index, exact, query, 10, allowed=allowed))) == sorted(farthest)
    assert allowed.lists == 1
    assert allowed.checked <= vectors.SCAN_SHARE * len(ids)  # given up on before checking more than a listing costs


def test_a_filter_over_several_collections_and_with_exclusions(monkeypatch):
    monkeypatch.setattr(vectors, "RESCORE", 5)  # checked, not listed, as in test_a_filter_letting_many_through...
    ids, matrix, exact, whole = random_index(n=6000)
    parts = [Codes(ids[i::3], codes(matrix[i::3])) for i in range(3)]
    for query in unit(np.random.default_rng(8).standard_normal((10, 64))):
        expected = top(whole, exact, query, 10, allowed=ids[::4], excluded=ids[:40])
        alike(top(parts, exact, query, 10, allowed=(allowed := Allowed(ids[::4])), excluded=ids[:40]), expected)
        assert allowed.lists == 0


def test_a_filter_letting_nothing_through():
    *_, exact, index = random_index(n=50)
    assert top(index, exact, unit(np.ones((1, 64)))[0], 5, allowed=Allowed([])) == []


def test_a_filter_over_no_codes_lists_nothing():
    # A collection not embedded yet: listing what the filter lets through read every document, to find nothing.
    allowed, query = Allowed([1, 2]), unit(np.ones((1, 64)))[0]
    assert top([Codes(np.empty(0, np.int64), codes(np.empty((0, 64))))], Exact([], []), query, 5, allowed=allowed) == []
    assert top([], Exact([], []), query, 5, allowed=allowed) == []
    assert allowed.lists == 0


def test_top_of_more_than_there_are_and_of_none():
    ids, matrix, exact, index = random_index(n=7)
    query = unit(np.ones((1, 64)))[0]
    assert chunks(top(index, exact, query, 50)) == exact_top(ids, matrix, query, 7)
    assert top(index, exact, query, 5, allowed=np.array([], np.int64)) == []
    assert top(Codes(np.empty(0, np.int64), codes(np.empty((0, 64), np.float16))), Exact([], []), query, 5) == []
    assert top([], Exact([], []), query, 5) == []


def test_vectors_gone_from_the_index_meanwhile_are_left_out():
    # A `sift index` running beside a search may delete vectors after the codes were loaded.
    ids, matrix, exact, index = random_index(n=50)
    query = unit(np.ones((1, 64)))[0]
    best = exact_top(ids, matrix, query, 3)
    del exact.rows[best[0]]
    assert chunks(top(index, exact, query, 2)) == best[1:]


def test_broken_vectors_are_left_out():
    # A vector the model broke (NaN) has no cosine: it must neither rank first, as PyTorch ranked NaN, nor take a place.
    ids, matrix, exact, index = random_index(n=20)
    query = unit(np.ones((1, 64)))[0]
    best = exact_top(ids, matrix, query, 3)
    exact.rows[best[0]] = np.full(64, np.nan, np.float16)
    assert chunks(top(index, exact, query, 2)) == best[1:]
    assert best[0] not in chunks(top(index, exact, query, 20))
    assert len(top(index, exact, query, 20)) == 19


def test_distances_split_over_threads_agree(monkeypatch):
    *_, exact, index = random_index(n=1000)
    query = unit(np.random.default_rng(4).standard_normal((1, 64)))[0]
    whole = top(index, exact, query, 20)
    monkeypatch.setattr(vectors, "BLOCK", 7)  # many blocks, each in a thread
    assert top(index, exact, query, 20) == whole


def test_dimensions_that_fill_no_whole_word():
    ids, matrix, exact, index = random_index(n=40, dim=4)
    query = unit(np.array([[1, 2, -1, 0.5]]))[0]
    assert chunks(top(index, exact, query, 5)) == exact_top(ids, matrix, query, 5)


# --- codes patched instead of loaded anew --------------------------------------


def test_patched_codes_find_new_vectors_and_forget_removed():
    # A note indexed again while the server runs: its old vectors go and its new ones come, without loading the
    # millions of others again.
    ids, matrix, exact, index = random_index(n=100)
    query = unit(np.ones((1, 64)))[0]
    best = exact_top(ids, matrix, query, 1)[0]
    new = unit(query[None, :] + 0.01).astype(np.float16)  # nearer the query than any other
    exact.rows[2000] = new[0]
    del exact.rows[best]
    patched = index.patched(removed=[best], ids=[2000], bits=codes(new))
    assert chunks(top(patched, exact, query, 2))[0] == 2000
    assert best not in chunks(top(patched, exact, query, 100))
    assert (len(index), 2000 in chunks(top(index, exact, query, 100))) == (
        100,
        False,
    )  # the codes patched from stay as were


def test_a_chunk_patched_in_again_is_its_new_vector_only():
    # A chunk embedded again is patched in under an id the codes hold already: it is the new vector of that chunk, not
    # a second one.
    ids, matrix, exact, index = random_index(n=100)
    query = unit(np.ones((1, 64)))[0]
    worst = exact_top(ids, matrix, -query, 1)[0]
    new = unit(query[None, :] + 0.01).astype(np.float16)
    exact.rows[worst] = new[0]
    patched = index.patched(removed=[], ids=[worst], bits=codes(new))
    ranked = chunks(top(patched, exact, query, 100))
    assert ranked[0] == worst
    assert ranked.count(worst) == 1
    again = patched.patched(removed=[worst], ids=[worst, 3000], bits=codes(np.vstack([new, new])))
    exact.rows[3000] = new[0]
    assert sorted(chunks(top(again, exact, query, 2))) == [worst, 3000]
    assert len(again) == 101


def test_codes_patched_many_times_hold_what_a_fresh_load_would():
    # A server runs for weeks and reindexes notes thousands of times: every patch must leave the codes a load from the
    # index would give, and add no part per patch.
    ids, matrix, _, index = random_index(n=50)
    rng, held = np.random.default_rng(5), dict(zip(ids.tolist(), codes(matrix)))
    for step in range(200):
        removed = rng.choice(list(held), size=rng.integers(0, 3), replace=False).tolist()
        added = rng.choice(80, size=rng.integers(0, 4), replace=False).tolist()  # ids embedded again too
        rows = codes(rng.standard_normal((len(added), 64)))
        index = index.patched(removed=removed, ids=added, bits=rows)
        for chunk in removed:
            held.pop(chunk)
        held.update(zip(added, rows))
        live_ids, live_bits = index.live()
        assert live_ids.tolist() == sorted(held), f"step {step}"
        assert (live_bits == np.array([held[c] for c in sorted(held)]).reshape(len(held), -1)).all(), f"step {step}"
        assert len(index) == len(held)
        assert len(index._parts) <= 2


# --- the codes of several collections ------------------------------------------


def test_the_codes_of_several_collections_are_searched_as_one():
    # Each collection's store has its codes: the candidates are the nearest of them all, as if they were one.
    ids, matrix, exact, whole = random_index(n=900)
    parts = [Codes(ids[i::3], codes(matrix[i::3])) for i in range(3)]
    for query in unit(np.random.default_rng(6).standard_normal((10, 64))):
        assert top(parts, exact, query, 10) == top(whole, exact, query, 10)
        assert top(parts, exact, query, 10, allowed=ids[::4]) == top(whole, exact, query, 10, allowed=ids[::4])


def test_few_candidates_are_the_nearest_of_all_collections(monkeypatch):
    # The candidates are the nearest of all collections together, not of each: a collection far from the query gives
    # none.
    monkeypatch.setattr(vectors, "RESCORE", 20)
    ids, matrix, exact, _ = random_index(n=400)
    query = unit(np.ones((1, 64)))[0]
    order = np.argsort(-(matrix.astype(np.float32) @ query))
    near, far = order[:200], order[200:]
    parts = [Codes(ids[near], codes(matrix[near])), Codes(ids[far], codes(matrix[far]))]
    top(parts, exact, query, 5)
    assert exact.asked == [20]


def test_a_scan_counts_the_codes_at_each_distance(monkeypatch):
    # Counted over all codes at once, by one thread, after the scan: half of a search over 46 million codes.
    monkeypatch.setattr(vectors, "BLOCK", 7)  # many blocks, each counted in its thread
    ids, matrix, _, index = random_index(n=100)
    query = codes(unit(np.ones((1, 64))))[0]
    expected = np.bitwise_count(codes(matrix) ^ query).sum(1)
    distances, counts = index.measure(query, excluded=ids[:10])
    assert counts.tolist() == np.bincount(expected[10:], minlength=8 * index.width + 1).tolist()  # those left out: none
    assert distances[10:].tolist() == expected[10:].tolist()
    assert (distances[:10] == vectors.FAR).all()


def test_the_codes_between_two_distances(monkeypatch):
    monkeypatch.setattr(vectors, "BLOCK", 7)
    ids, matrix, _, index = random_index(n=100)
    patched = index.patched(removed=ids[:5], ids=[5000, 5010], bits=codes(matrix[:2]))  # in two parts
    distances, _ = patched.measure(codes(unit(np.ones((1, 64))))[0])
    for low, high in ((0, 30), (25, 33), (40, 20), (0, 64)):
        expected = np.flatnonzero((distances >= low) & (distances <= high))
        assert patched.between(distances, low, high).tolist() == expected.tolist()


def test_the_distances_of_one_collections_codes():
    ids, matrix, _, index = random_index(n=100)
    query = codes(unit(np.ones((1, 64))))[0]
    expected = np.bitwise_count(codes(matrix) ^ query).sum(1)
    assert index.distances(query).tolist() == expected.tolist()  # by id, as the codes hold them
    left = index.distances(query, allowed=ids[:50], excluded=ids[:10])
    assert (left[10:50] == expected[10:50]).all()
    assert (left[:10] == vectors.FAR).all()
    assert (left[50:] == vectors.FAR).all()
