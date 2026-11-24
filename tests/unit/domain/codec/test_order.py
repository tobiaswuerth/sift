import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.arrays import Ids, ints
from sift.domain.codec.codes import Bits, code_bytes
from sift.domain.codec.order import topical

WIDTH = 256
FLIPPED = 0.01


def random_codes(seed: int, count: int, width: int = WIDTH) -> Bits:
    return np.random.default_rng(seed).integers(0, 256, (count, code_bytes(width)), np.uint8)


def near(code: Bits, seed: int, count: int) -> Bits:
    held = np.unpackbits(np.repeat(code, count, axis=0), axis=1)
    noise = (np.random.default_rng(seed).random(held.shape) < FLIPPED).astype(np.uint8)
    return np.packbits(held ^ noise, axis=1)


def runs_of(order: Ids, topic: dict[int, int]) -> list[int]:
    topics = [topic[each] for each in ints(order)]
    return [each for at, each in enumerate(topics) if at == 0 or topics[at - 1] != each]


@given(st.integers(0, 400), st.integers(0, 2**32 - 1), st.integers(1, 2**40))
def test_the_order_is_every_chunk_once_and_the_same_each_time(count: int, seed: int, start: int) -> None:
    ids = np.sort(np.random.default_rng(seed).choice(np.arange(start, start + 4 * count + 1), count, replace=False))
    codes = random_codes(seed, count)
    order = topical(ids, codes)
    assert sorted(ints(order)) == ints(ids)
    assert ints(order) == ints(topical(ids, codes))


def test_chunks_of_one_topic_lie_together_however_their_ids_mix() -> None:
    topics = random_codes(7, 5)
    codes = np.concatenate([near(topics[at : at + 1], at, 60) for at in range(len(topics))])
    shuffled = np.random.default_rng(3).permutation(len(codes))
    ids = np.arange(1000, 1000 + len(codes), dtype=np.int64)
    topic = {chunk: at // 60 for chunk, at in zip(ints(ids), ints(shuffled), strict=True)}
    order = topical(ids, codes[shuffled])
    assert sorted(runs_of(order, topic)) == [0, 1, 2, 3, 4]


def test_codes_measured_a_part_at_a_time_give_the_order_measured_at_once() -> None:
    topics = random_codes(11, 6)
    codes = np.concatenate([near(topics[at : at + 1], at, 300) for at in range(len(topics))])
    ids = np.arange(len(codes), dtype=np.int64)
    assert ints(topical(ids, codes, step=64)) == ints(topical(ids, codes))


def test_chunks_of_one_code_keep_the_order_of_their_ids() -> None:
    code = random_codes(1, 1)
    ids = np.array([5, 9, 12, 40], np.int64)
    assert ints(topical(ids, np.repeat(code, 4, axis=0))) == [5, 9, 12, 40]
    assert ints(topical(ids[:1], code)) == [5]
    assert ints(topical(ids[:0], code[:0])) == []
