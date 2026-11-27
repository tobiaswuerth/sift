import json
from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from sift.domain.arrays import floats, ints
from sift.domain.codec.postings import MOST_INFLATED_BYTES, blocked, lengths_of, positions_of, ranges_of
from sift.domain.codec.segments import (
    GAP,
    LongPostings,
    Postings,
    Segment,
    ShortPostings,
    building,
    stream,
    written,
)
from sift.domain.codec.shape import MOST_BLOCK, SHAPE, Shape
from sift.domain.rank.bm25 import term_scores, unquantized
from sift.domain.text.tokens import MOST_FIELD_CHARACTERS, field_tokens, tokens

SMALL = Shape(range_shift=2, block=2, long=3, lengths_shift=3)
MOST_BYTES = 9
MOST_TIMES = 1024
EVERY_TOKEN = 1 << 62

type Columns = tuple[list[str], ...]
words = st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), max_size=6)
chunks = st.lists(st.tuples(words, words, words), min_size=1, max_size=25)


def segment(ids: NDArray[np.int64], columns: Sequence[Columns], average: float, shape: Shape = SHAPE) -> Segment:
    built = building(EVERY_TOKEN)
    fields = [(" ".join(title), " ".join(meta)) for title, _, meta in columns]
    assert built.add(ids, range(len(columns)), fields, [" ".join(body) for _, body, _ in columns]) == len(columns)
    return written(built, average, shape)


def decoded(held: Postings) -> tuple[list[int], list[int], list[int]]:
    parts = held.held.blocks if isinstance(held.held, LongPostings) else (held.held,)
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
    made = segment(ids, columns, average=4.0, shape=shape)
    want = {
        term: [
            (listed[n], [p for p, token in enumerate(s) if token == term]) for n, s in enumerate(streams) if term in s
        ]
        for term in {token for s in streams for token in s} - {GAP}
    }
    got = {held.term: decoded(held) for held in made.postings}
    assert got.keys() == want.keys()
    for term, (found, counts, positions) in got.items():
        assert found == [chunk for chunk, _ in want[term]]
        assert counts == [len(places) for _, places in want[term]]
        assert positions == [p for _, places in want[term] for p in places]
    assert {held.term: held.chunks for held in made.postings} == {term: len(each) for term, each in want.items()}
    lengths = {listed[n]: sum(token != GAP for token in s) for n, s in enumerate(streams)}
    assert made.tokens == sum(lengths.values())
    assert (made.first, made.last, made.chunks) == (listed[0], listed[-1], len(ids))
    for block, data in made.lengths.items():
        dense = lengths_of(data, shape.lengths_shift)
        for chunk, length in lengths.items():
            if chunk >> shape.lengths_shift == block:
                assert dense[chunk & ((1 << shape.lengths_shift) - 1)] == length


@given(chunks)
def test_the_ranges_of_a_long_list_bound_every_score_in_them(columns: list[tuple[list[str], ...]]) -> None:
    ids = np.arange(len(columns), dtype=np.int64) + 100
    streams = [stream(*each) for each in columns]
    lengths = np.array([sum(token != GAP for token in s) for s in streams])
    for held in segment(ids, columns, average=3.0, shape=SMALL).postings:
        if not isinstance(held.held, LongPostings):
            assert isinstance(held.held, ShortPostings)
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


@pytest.mark.parametrize(
    "asked, refusal",
    [
        ({"range_shift": 9}, "its counts are bytes"),
        ({"range_shift": -1}, "its counts are bytes"),
        ({"block": 0}, "one entry at least"),
        ({"lengths_shift": 0}, "one entry at least"),
        ({"block": (1 << 12) + 1}, "at most 4,096"),
        ({"long": (1 << 12) + 1}, "at most 4,096"),
        ({"lengths_shift": 21}, r"2\*\*20"),
    ],
)
def test_a_shape_whose_ranges_or_blocks_cannot_be_written_is_refused(asked: dict[str, int], refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        Shape(**asked)


def test_a_word_counts_in_a_passage_no_more_often_than_a_list_may_hold_and_the_passage_stays_as_long() -> None:
    often = MOST_TIMES + 500
    columns: list[Columns] = [([], ["x"] * often, ["y"]), (["x"], [], [])]
    made = segment(np.array([7, 9], np.int64), columns, average=4.0)
    held = {each.term: decoded(each) for each in made.postings}
    assert held["x"] == ([7, 9], [MOST_TIMES, 1], [*range(1, MOST_TIMES + 1), 0])
    assert held["y"] == ([7], [1], [often + 2])
    assert made.tokens == often + 2


def test_no_list_a_shape_lets_be_written_unpacks_to_more_than_a_reader_takes() -> None:
    assert MOST_BLOCK * MOST_TIMES * MOST_BYTES <= MOST_INFLATED_BYTES


def test_a_chunk_of_no_word_is_in_the_segment_with_no_tokens() -> None:
    made = segment(np.array([7, 8], np.int64), [([], [], []), (["a"], [], [])], average=0.5)
    assert [(held.term, held.chunks) for held in made.postings] == [("a", 1)]
    assert ints(lengths_of(made.lengths[0], 10)[7:9]) == [0, 1]


@given(st.lists(st.text(st.characters(exclude_categories=["Cs"]), max_size=40), min_size=1, max_size=8))
def test_a_segment_takes_the_words_of_its_texts_as_a_query_s_words_are_taken(texts: list[str]) -> None:
    built = building(EVERY_TOKEN)
    ids = np.arange(len(texts), dtype=np.int64) + 5
    assert built.add(ids, range(len(texts)), [("", "")] * len(texts), texts) == len(texts)
    words = [tokens(each) for each in texts]
    if not any(words):
        assert written(built, 1.0).postings == ()
        return
    got = {held.term: decoded(held) for held in written(built, 1.0).postings}
    want = {
        term: [
            (5 + n, [1 + p for p, word in enumerate(held) if word == term])
            for n, held in enumerate(words)
            if term in held
        ]
        for term in {word for held in words for word in held}
    }
    assert list(got) == sorted(want)
    for term, (found, counts, positions) in got.items():
        assert (found, counts) == ([chunk for chunk, _ in want[term]], [len(places) for _, places in want[term]])
        assert positions == [p for _, places in want[term] for p in places]


def test_a_segment_takes_passages_until_it_holds_the_tokens_asked_the_one_that_reaches_them_too() -> None:
    built = building(9)
    ids = np.arange(4, dtype=np.int64)
    assert built.add(ids[:1], [0], [("a", "b")], ["c d"]) == 1
    assert (built.chunks, built.tokens) == (1, 4)
    assert built.add(ids[1:], [0, 0, 0], [("", "")], ["e", "f", "g"]) == 1
    assert (built.chunks, built.tokens) == (2, 5)
    assert built.add(ids[2:], [0, 0], [("", "")], ["f", "g"]) == 0
    assert [held.term for held in written(built, 1.0).postings] == ["a", "b", "c", "d", "e"]


def test_a_title_or_fields_longer_than_sift_reads_lose_the_word_they_were_cut_in() -> None:
    whole, cut = "a " * (MOST_FIELD_CHARACTERS // 2), "a " * (MOST_FIELD_CHARACTERS // 2) + "bb"
    built = building(EVERY_TOKEN)
    assert built.add(np.arange(2, dtype=np.int64), [0, 1], [(whole, ""), (cut, cut)], ["", ""]) == 2
    (held,) = written(built, 1.0).postings
    ids, counts, _ = decoded(held)
    assert (held.term, ids) == ("a", [0, 1])
    assert counts == [MOST_TIMES, MOST_TIMES]
    assert field_tokens(cut) == ["a"] * (MOST_FIELD_CHARACTERS // 2 - 1)


def test_a_segment_is_written_once_and_refuses_what_it_was_not_given_whole() -> None:
    built = building(EVERY_TOKEN)
    one = np.zeros(1, np.int64)
    with pytest.raises(ValueError, match="names a document that was not given"):
        built.add(one, [1], [("", "")], ["a"])
    with pytest.raises(ValueError, match="a place and a text"):
        built.add(one, [0], [("", "")], ["a", "b"])
    with pytest.raises(ValueError, match="holds a chunk at least"):
        written(building(EVERY_TOKEN), 1.0)
    assert built.add(one, [0], [("", "")], ["a"]) == 1
    assert written(built, 1.0).chunks == 1
    for used in (lambda: built.add(one, [0], [("", "")], ["a"]), lambda: written(built, 1.0)):
        with pytest.raises(ValueError, match="written once"):
            used()


@pytest.mark.parametrize("written", ['{"range_shift": 7.5}', '{"block": true}', '{"long": "9"}'])
def test_a_shape_of_other_than_whole_numbers_is_refused(written: str) -> None:
    with pytest.raises(TypeError, match="whole numbers"):
        Shape(**cast("dict[str, int]", json.loads(written)))
