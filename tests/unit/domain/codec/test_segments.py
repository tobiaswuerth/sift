import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.arrays import floats, ints
from sift.domain.codec.postings import blocked, lengths_of, positions_of, ranges_of
from sift.domain.codec.segments import GAP, Long, Postings, Shape, Short, segment, stream
from sift.domain.rank.bm25 import term_scores, unquantized

SMALL = Shape(range_shift=2, block=2, long=3, lengths_shift=3)
words = st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), max_size=6)
chunks = st.lists(st.tuples(words, words, words), min_size=1, max_size=25)


def decoded(held: Postings) -> tuple[list[int], list[int], list[int]]:
    parts = held.held.blocks if isinstance(held.held, Long) else (held.held,)
    ids: list[int] = []
    counts: list[int] = []
    positions: list[int] = []
    for part in parts:
        found, tf = blocked(part.postings)
        ids += ints(found)
        counts += ints(tf)
        positions += ints(positions_of(part.positions, tf))
    return ids, counts, positions


@given(chunks, st.integers(0, 1 << 40), st.sampled_from([SMALL, Shape()]))
def test_a_segment_holds_where_each_term_occurs(columns: list[tuple[list[str], ...]], first: int, shape: Shape) -> None:
    ids = first + 3 * np.arange(len(columns), dtype=np.int64)
    listed = ints(ids)
    streams = [stream(*each) for each in columns]
    written = segment(ids, streams, average=4.0, shape=shape)
    want = {
        term: [
            (listed[n], [p for p, token in enumerate(s) if token == term]) for n, s in enumerate(streams) if term in s
        ]
        for term in {token for s in streams for token in s} - {GAP}
    }
    got = {held.term: decoded(held) for held in written.postings}
    assert got.keys() == want.keys()
    for term, (found, counts, positions) in got.items():
        assert found == [chunk for chunk, _ in want[term]]
        assert counts == [len(places) for _, places in want[term]]
        assert positions == [p for _, places in want[term] for p in places]
    assert {held.term: held.chunks for held in written.postings} == {term: len(each) for term, each in want.items()}
    lengths = {listed[n]: sum(token != GAP for token in s) for n, s in enumerate(streams)}
    assert written.tokens == sum(lengths.values())
    assert (written.first, written.last, written.chunks) == (listed[0], listed[-1], len(ids))
    for block, data in written.lengths.items():
        dense = lengths_of(data)
        for chunk, length in lengths.items():
            if chunk >> shape.lengths_shift == block:
                assert dense[chunk & ((1 << shape.lengths_shift) - 1)] == length


@given(chunks)
def test_the_ranges_of_a_long_list_bound_every_score_in_them(columns: list[tuple[list[str], ...]]) -> None:
    ids = np.arange(len(columns), dtype=np.int64) + 100
    streams = [stream(*each) for each in columns]
    lengths = np.array([sum(token != GAP for token in s) for s in streams])
    for held in segment(ids, streams, average=3.0, shape=SMALL).postings:
        if not isinstance(held.held, Long):
            assert isinstance(held.held, Short)
            assert held.chunks < SMALL.long
            continue
        assert held.chunks >= SMALL.long
        found, counts, _ = decoded(held)
        scores = term_scores(np.array(counts), lengths[np.array(found) - 100], 3.0)
        ranges = ranges_of(held.held.ranges)
        bests = floats(unquantized(ranges.best))
        for rid, start, end, best in zip(ints(ranges.ids), ints(ranges.starts), ints(ranges.ends), bests, strict=True):
            assert all(chunk >> SMALL.range_shift == rid for chunk in found[start:end])
            assert best >= scores[start:end].max()
        assert ranges.ends[-1] == len(found)


def test_a_chunk_of_no_word_is_in_the_segment_with_no_tokens() -> None:
    written = segment(np.array([7, 8], np.int64), [stream([], [], []), stream(["a"], [], [])], average=0.5)
    assert [(held.term, held.chunks) for held in written.postings] == [("a", 1)]
    assert ints(lengths_of(written.lengths[0])[7:9]) == [0, 1]
