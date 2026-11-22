import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.arrays import ints
from sift.domain.rank.phrases import Occurrences, merged, phrase_counts

texts = st.lists(st.lists(st.sampled_from("abc"), max_size=12), min_size=1, max_size=8)


def occurrences(chunks: list[list[str]], token: str) -> Occurrences:
    held = [(n, [p for p, t in enumerate(text) if t == token]) for n, text in enumerate(chunks)]
    held = [(n, places) for n, places in held if places]
    return (
        np.array([10 * n for n, _ in held], np.int64),
        np.array([len(places) for _, places in held], np.int64),
        np.array([p for _, places in held for p in places], np.int64),
    )


@given(texts, st.lists(st.sampled_from("abc"), min_size=2, max_size=4))
def test_a_phrase_is_counted_where_its_tokens_follow_each_other_overlapping_ones_each(
    chunks: list[list[str]], phrase: list[str]
) -> None:
    candidates = np.array([10 * n for n in range(len(chunks))], np.int64)
    want = [sum(text[at : at + len(phrase)] == phrase for at in range(len(text))) for text in chunks]
    got = phrase_counts(candidates, [occurrences(chunks, token) for token in phrase])
    assert got.tolist() == want


def test_positions_far_apart_are_counted_without_overflow() -> None:
    far = 1 << 61
    candidates = np.arange(0, 40, 10, dtype=np.int64)
    first = (candidates, np.ones(4, np.int64), np.full(4, far, np.int64))
    second = (candidates, np.ones(4, np.int64), np.full(4, far + 1, np.int64))
    assert ints(phrase_counts(candidates, [first, second])) == [1, 1, 1, 1]


def test_many_candidates_are_counted_without_overflow() -> None:
    candidates = np.arange(1 << 20, dtype=np.int64)
    half = candidates[::2]
    first = (candidates, np.ones(len(candidates), np.int64), np.full(len(candidates), 1 << 44, np.int64))
    second = (half, np.ones(len(half), np.int64), np.full(len(half), (1 << 44) + 1, np.int64))
    counts = phrase_counts(candidates, [first, second])
    assert int(counts.sum()) == len(half)
    assert ints(counts[:4]) == [1, 0, 1, 0]


@given(st.lists(st.tuples(st.integers(0, 20), st.integers(1, 5)), max_size=40))
def test_postings_merged_sum_their_counts_by_place(pairs: list[tuple[int, int]]) -> None:
    places, counts = merged(np.array([p for p, _ in pairs], np.int64), np.array([c for _, c in pairs], np.int64))
    want: dict[int, int] = {}
    for place, count in pairs:
        want[place] = want.get(place, 0) + count
    assert dict(zip(ints(places), ints(counts), strict=True)) == want
    assert places.tolist() == sorted(want)
