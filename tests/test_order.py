"""The topical order of a collection's chunks in its keyword postings (sift.order), from their vectors' codes."""

import numpy as np
import pytest

from sift import order


def clustered(n_clusters, size, rng, flip=0.05):
    """Codes of `n_clusters` topics, `size` chunks each, every chunk its topic's bits with a few flipped, and the
    chunks' ids shuffled: (ids, codes, topic of each)."""
    topics = rng.integers(0, 2, (n_clusters, 768), dtype=np.uint8)
    topic = np.repeat(np.arange(n_clusters), size)
    bits = topics[topic] ^ (rng.random((len(topic), 768)) < flip).astype(np.uint8)
    ids = rng.permutation(len(topic)).astype(np.int64) + (3 << 40)
    return ids, np.packbits(bits, axis=1), topic


def runs(values):
    return 1 + int(np.count_nonzero(np.diff(values)))


def test_chunks_of_a_topic_come_together(monkeypatch):
    monkeypatch.setattr(order, "LEVELS", (8, 4))
    ids, bits, topic = clustered(6, 300, np.random.default_rng(1))
    ordered = order.topical(ids, bits)
    assert sorted(ordered.tolist()) == sorted(ids.tolist())  # each chunk once
    of = dict(zip(ids.tolist(), topic.tolist()))
    assert runs([of[i] for i in ordered.tolist()]) == 6  # each topic in one run


def test_a_cluster_keeps_its_ids_ascending(monkeypatch):
    # Within a leaf the ids stay in order: their gaps are smaller, and compress better.
    monkeypatch.setattr(order, "LEVELS", (6,))
    ids, bits, topic = clustered(6, 200, np.random.default_rng(2))
    of = dict(zip(ids.tolist(), topic.tolist()))
    ordered = order.topical(ids, bits).tolist()
    for t in range(6):
        mine = [i for i in ordered if of[i] == t]
        assert mine == sorted(mine)


def test_the_same_codes_give_the_same_order():
    ids, bits, _ = clustered(5, 100, np.random.default_rng(3))
    assert order.topical(ids, bits).tolist() == order.topical(ids, bits).tolist()


@pytest.mark.parametrize("n", [0, 1, 2, 7])
def test_few_chunks(n):
    rng = np.random.default_rng(4)
    ids = np.arange(n, dtype=np.int64) + 10
    bits = rng.integers(0, 256, (n, 96), dtype=np.uint8)
    assert sorted(order.topical(ids, bits).tolist()) == ids.tolist()


def test_large_inputs_are_measured_in_parts(monkeypatch):
    # Distances are measured a part of the codes at a time, in threads: the same order as in one part.
    monkeypatch.setattr(order, "LEVELS", (8, 4))
    ids, bits, _ = clustered(6, 300, np.random.default_rng(5))
    whole = order.topical(ids, bits).tolist()
    monkeypatch.setattr(order, "STEP", 64)
    assert order.topical(ids, bits).tolist() == whole
