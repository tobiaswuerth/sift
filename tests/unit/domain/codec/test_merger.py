import contextlib
import itertools
from collections.abc import Iterable, Iterator, Sequence

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from sift.domain.codec.segments import (
    GAP,
    FiledLists,
    ListBlockRow,
    MergedChunks,
    MergeLimits,
    Merger,
    MergeStep,
    WrittenSegment,
    fed,
    fed_packs,
    merger,
    stepped,
    stream,
)
from sift.domain.codec.shape import Shape
from sift.domain.errors import DamagedError
from tests.coded_lists import block, pack, placed
from tests.unit.domain.codec.built import SEGMENT, SMALL, Columns, chunks, of_bodies, segment

EVERYTHING = 10**9
WHOLE = MergeLimits(postings=EVERYTHING, stored=EVERYTHING, tokens=EVERYTHING)
PIECES = MergeLimits(postings=1, stored=1, tokens=EVERYTHING)
DENSE = MergeLimits(postings=EVERYTHING, stored=EVERYTHING, tokens=1)
AVERAGE = 4.0
FEEDS = (fed_packs, fed)

type Source = list[tuple[object, ...]]


def sources(made: WrittenSegment, shape: Shape) -> list[Source]:
    packs: Source = [(lists, places) for _, _, lists, places in made.filed.packs]
    blocks: Source = [(term, shape.block, coded, positions) for _, term, _, coded, positions in made.filed.blocks]
    return [packs, blocks]


def steps(held: Merger, given: Sequence[Source], at_once: int) -> Iterator[tuple[MergeStep, int]]:
    left = [itertools.batched(each, at_once) for each in given]
    taken, done = 0, False
    while not done:
        step = stepped(held, SEGMENT)
        yield step, taken
        for at in step.wanted:
            more = True
            while more:
                rows = list(next(left[at], ()))
                more = FEEDS[at % len(FEEDS)](held, at, rows)
                taken += len(rows)
        done = step.done


def whole(ids: Sequence[int]) -> NDArray[np.int64]:
    return np.array(ids, np.int64)


def numbered(row: ListBlockRow) -> tuple[str, int]:
    return row[1], row[2]


def filed(made: Iterable[MergeStep]) -> FiledLists:
    each = [step.filed for step in made]
    blocks = sorted((row for one in each for row in one.blocks), key=numbered)
    return FiledLists([row for one in each for row in one.packs], [row for one in each for row in one.ranged], blocks)


@given(
    st.lists(chunks, min_size=1, max_size=4),
    st.data(),
    st.sampled_from([SMALL, Shape()]),
    st.sampled_from([WHOLE, PIECES, DENSE]),
    st.sampled_from([1, 2, 64]),
)
def test_segments_merge_into_the_segment_their_live_chunks_alone_build(
    parts: list[list[Columns]], data: st.DataObject, shape: Shape, limits: MergeLimits, at_once: int
) -> None:
    columns = [chunk for part in parts for chunk in part]
    ids = 5 + 3 * np.arange(len(columns), dtype=np.int64)
    dead = sorted(data.draw(st.sets(st.integers(1, len(columns) - 1))) if len(columns) > 1 else set[int]())
    live = [n for n in range(len(columns)) if n not in dead]
    given = [
        source
        for start, end in itertools.pairwise(itertools.accumulate(map(len, parts), initial=0))
        for source in sources(segment(ids[start:end], columns[start:end], AVERAGE, shape), shape)
    ]
    lengths = whole([sum(token != GAP for token in stream(*columns[n])) for n in live])
    held = merger(MergedChunks(ids[whole(live)], lengths, ids[whole(dead)], AVERAGE), shape, limits, len(given))
    want = segment(ids[whole(live)], [columns[n] for n in live], AVERAGE, shape)
    assert filed(step for step, _ in steps(held, given, at_once)) == want.filed


def chunks_of(ids: range) -> MergedChunks:
    return MergedChunks(whole(ids), whole([2] * len(ids)), whole([]), AVERAGE)


def word(number: int) -> str:
    return "".join("abcdefghij"[int(digit)] for digit in f"{number:05}")


def test_a_merge_asks_for_the_rows_of_a_source_once_it_holds_few_of_them_however_many_there_are() -> None:
    at_once, terms, a_chunk, ahead = 4, 20_000, 100, 4096
    made = [
        segment(
            start * terms + np.arange(terms // a_chunk, dtype=np.int64),
            of_bodies([word(2 * n + start) for n in range(at, at + a_chunk)] for at in range(0, terms, a_chunk)),
            AVERAGE,
        )
        for start in (0, 1)
    ]
    given = [source for each in made for source in sources(each, Shape())]
    a_pack = terms // len(made[0].filed.packs)
    ids = whole([*range(terms // a_chunk), *range(terms, terms + terms // a_chunk)])
    limits = MergeLimits(len(given) * ahead, EVERYTHING, EVERYTHING)
    held = merger(MergedChunks(ids, whole([a_chunk] * len(ids)), whole([]), AVERAGE), Shape(), limits, len(given))
    most = 2 * (ahead // a_pack + 1 + at_once) + 4
    written, deepest = 0, 0
    for step, taken in steps(held, given, at_once):
        deepest = max(deepest, taken - written)
        written += len(step.filed.packs)
    assert 2 * (ahead // a_pack) <= deepest <= most < len(made[0].filed.packs)
    assert written >= 2 * len(made[0].filed.packs) - 1


def test_a_term_past_what_a_merge_takes_at_once_comes_back_in_blocks_as_its_rows_are_fed() -> None:
    a_segment = 40
    given = [
        source
        for first in (0, a_segment)
        for source in sources(
            segment(first + np.arange(a_segment, dtype=np.int64), of_bodies([["w"]] * a_segment), AVERAGE, SMALL), SMALL
        )
    ]
    held = merger(chunks_of(range(2 * a_segment)), SMALL, PIECES, len(given))
    made = [
        (len(step.filed.blocks), [term for _, term, *_ in step.filed.ranged], taken)
        for step, taken in steps(held, given, 1)
    ]
    written = list(itertools.accumulate(blocks for blocks, _, _ in made))
    assert all(taken - before <= 4 for (_, _, taken), before in zip(made, written, strict=True))
    assert [terms for _, terms, _ in made if terms] == [["w"]]
    assert written[[terms for _, terms, _ in made].index(["w"])] == written[-1] == a_segment


def thrice(words: str, first: int) -> list[Source]:
    return sources(segment(first + np.arange(3, dtype=np.int64), of_bodies([words.split()] * 3), AVERAGE), Shape())


def test_a_merge_of_lists_that_are_damaged_or_hold_what_the_store_does_not_is_damage() -> None:
    given = thrice("a", 0)
    held = merger(chunks_of(range(1, 3)), Shape(), WHOLE, len(given))
    with pytest.raises(DamagedError, match="chunks the store does not"):
        list(steps(held, given, 1))
    held = merger(chunks_of(range(3)), Shape(), WHOLE, len(given))
    fed(held, 1, [("a", 3, b"\x00", b"\x00")])
    fed(held, 1, [])
    fed_packs(held, 0, [])
    with pytest.raises(DamagedError, match="shorter than its head"):
        stepped(held, SEGMENT)
    with pytest.raises(DamagedError, match="a pack of lists holds other than its head says"):
        fed_packs(held, 0, [(b"\x09", b"")])
    with pytest.raises(DamagedError, match="is no pack of lists"):
        fed_packs(held, 0, [(1, b"")])
    damaged: list[tuple[object, ...]] = [
        (b"a", 3, b"", b""),
        ("a", 3, None, b""),
        ("a", "3", b"", b""),
        ("a", -1, b"", b""),
    ]
    for row in damaged:
        with pytest.raises(DamagedError, match="is no list of a term"):
            fed(held, 1, [row])


plausible = st.lists(st.integers(0, 8), min_size=1, max_size=6, unique=True).flatmap(
    lambda ids: st.tuples(
        st.just(sorted(ids)),
        st.lists(st.integers(1, 3), min_size=len(ids), max_size=len(ids)),
        st.lists(
            st.lists(st.integers(0, 9), max_size=4, unique=True).map(sorted), min_size=len(ids), max_size=len(ids)
        ),
    )
)
hostile = st.one_of(
    st.tuples(st.binary(max_size=40), st.binary(max_size=40)),
    plausible.map(lambda held: (block(held[0], held[1]), placed(held[2]))),
    plausible.map(lambda held: (block(held[0], [len(each) or 1 for each in held[2]]), placed(held[2]))),
)


@given(
    st.dictionaries(st.sampled_from(["a", "b", "c"]), st.tuples(st.integers(0, 9), hostile), max_size=3),
    st.lists(st.tuples(st.sampled_from(["a", "b", "c"]), st.integers(0, 9), hostile), max_size=4),
    st.one_of(st.none(), st.tuples(st.binary(max_size=30), st.binary(max_size=30))),
    st.sampled_from([WHOLE, PIECES, DENSE]),
    st.sampled_from([SMALL, Shape()]),
)
def test_rows_that_hold_no_sound_lists_merge_or_are_damage_and_nothing_else(
    short: dict[str, tuple[int, tuple[bytes, bytes]]],
    blocks: list[tuple[str, int, tuple[bytes, bytes]]],
    junk: tuple[bytes, bytes] | None,
    limits: MergeLimits,
    shape: Shape,
) -> None:
    held = merger(chunks_of(range(3)), shape, limits, 2)
    packs = [pack([(term, count, *coded) for term, (count, coded) in sorted(short.items())])] if short else []
    with contextlib.suppress(DamagedError):
        fed_packs(held, 0, [*packs, *([junk] if junk else [])])
        fed_packs(held, 0, [])
        fed(held, 1, [(term, count, *coded) for term, count, coded in sorted(blocks)])
        fed(held, 1, [])
        assert any(stepped(held, SEGMENT).done for _ in range(len(short) + len(blocks) + 2))


def test_a_merge_takes_rows_of_the_sources_it_was_made_for_alone() -> None:
    held = merger(chunks_of(range(3)), Shape(), WHOLE, 2)
    for feed, rows in ((fed, [("a", 3, b"", b"")]), (fed, []), (fed_packs, [(b"", b"")]), (fed_packs, [])):
        with pytest.raises(IndexError, match="a source it has not"):
            feed(held, 2, rows)
