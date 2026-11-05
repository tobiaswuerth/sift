"""The index's vectors in memory as one bit per dimension, its sign: candidates by Hamming distance, the best of them
scored exactly with the vectors the index holds."""

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
    return ids, matrix, exact, Codes(ids, codes(matrix), exact)


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
    ids, matrix, _, index = random_index()
    for query in unit(np.random.default_rng(2).standard_normal((20, 64))):
        ranked = index.top(query, 10)
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
        assert chunks(index.top(query, 1)) == [ids[i]]
    assert exact.asked == [100] * 30  # only the candidates are read from the index


def test_top_keeps_to_the_allowed_chunks_and_leaves_out_the_excluded():
    ids, matrix, _, index = random_index()
    query = unit(np.ones((1, 64)))[0]
    allowed = ids[::3]
    ranked = index.top(query, 5, allowed=allowed, excluded=allowed[:2])
    keep = set(allowed.tolist()) - set(allowed[:2].tolist())
    assert chunks(ranked) == exact_top(ids, matrix, query, 5, keep)


def test_allowed_chunks_far_from_the_query_are_still_found(monkeypatch):
    # A filter applies before the candidates are picked: a small collection among millions isn't crowded out.
    monkeypatch.setattr(vectors, "RESCORE", 10)
    ids, matrix, _, index = random_index(n=2000)
    query = unit(np.ones((1, 64)))[0]
    farthest = [int(ids[i]) for i in np.argsort(matrix.astype(np.float32) @ query)[:3]]
    assert sorted(chunks(index.top(query, 10, allowed=np.array(farthest)))) == sorted(farthest)


def test_top_of_more_than_there_are_and_of_none():
    ids, matrix, _, index = random_index(n=7)
    query = unit(np.ones((1, 64)))[0]
    assert chunks(index.top(query, 50)) == exact_top(ids, matrix, query, 7)
    assert index.top(query, 5, allowed=np.array([], np.int64)) == []
    assert Codes(np.empty(0, np.int64), codes(np.empty((0, 64), np.float16)), Exact([], [])).top(query, 5) == []


def test_vectors_gone_from_the_index_meanwhile_are_left_out():
    # A `sift index` running beside a search may delete vectors after the codes were loaded.
    ids, matrix, exact, index = random_index(n=50)
    query = unit(np.ones((1, 64)))[0]
    best = exact_top(ids, matrix, query, 3)
    del exact.rows[best[0]]
    assert chunks(index.top(query, 2)) == best[1:]


def test_broken_vectors_are_left_out():
    # A vector the model broke (NaN) has no cosine: it must neither rank first, as PyTorch ranked NaN, nor take a place.
    ids, matrix, exact, index = random_index(n=20)
    query = unit(np.ones((1, 64)))[0]
    best = exact_top(ids, matrix, query, 3)
    exact.rows[best[0]] = np.full(64, np.nan, np.float16)
    assert chunks(index.top(query, 2)) == best[1:]
    assert best[0] not in chunks(index.top(query, 20))
    assert len(index.top(query, 20)) == 19


def test_distances_split_over_threads_agree(monkeypatch):
    *_, index = random_index(n=1000)
    query = unit(np.random.default_rng(4).standard_normal((1, 64)))[0]
    whole = index.top(query, 20)
    monkeypatch.setattr(vectors, "BLOCK", 7)  # many blocks, each in a thread
    assert index.top(query, 20) == whole


def test_dimensions_that_fill_no_whole_word():
    ids, matrix, _, index = random_index(n=40, dim=4)
    query = unit(np.array([[1, 2, -1, 0.5]]))[0]
    assert chunks(index.top(query, 5)) == exact_top(ids, matrix, query, 5)


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
    assert chunks(patched.top(query, 2))[0] == 2000
    assert best not in chunks(patched.top(query, 100))
    assert (len(index), 2000 in chunks(index.top(query, 100))) == (100, False)  # the codes patched from stay as were


def test_a_chunk_patched_in_again_is_its_new_vector_only():
    # SQLite hands out the ids of deleted chunks again: a chunk patched in under an id the codes hold already is the
    # new vector of that chunk, not a second one.
    ids, matrix, exact, index = random_index(n=100)
    query = unit(np.ones((1, 64)))[0]
    worst = exact_top(ids, matrix, -query, 1)[0]
    new = unit(query[None, :] + 0.01).astype(np.float16)
    exact.rows[worst] = new[0]
    patched = index.patched(removed=[], ids=[worst], bits=codes(new))
    ranked = chunks(patched.top(query, 100))
    assert ranked[0] == worst
    assert ranked.count(worst) == 1
    again = patched.patched(removed=[worst], ids=[worst, 3000], bits=codes(np.vstack([new, new])))
    exact.rows[3000] = new[0]
    assert sorted(chunks(again.top(query, 2))) == [worst, 3000]
    assert len(again) == 101


def test_codes_patched_many_times_hold_what_a_fresh_load_would():
    # A server runs for weeks and reindexes notes thousands of times: every patch must leave the codes a load from the
    # index would give, and add no part per patch.
    ids, matrix, _, index = random_index(n=50)
    rng, held = np.random.default_rng(5), dict(zip(ids.tolist(), codes(matrix)))
    for step in range(200):
        removed = rng.choice(list(held), size=rng.integers(0, 3), replace=False).tolist()
        added = rng.choice(80, size=rng.integers(0, 4), replace=False).tolist()  # ids taken again, as SQLite does
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
