from collections import Counter
from typing import NamedTuple

import numpy as np
from hypothesis import strategies as st

from sift.domain.arrays import ints
from sift.domain.codec.postings import blocked, positions_of
from sift.domain.plan.segments import Held, Policy, merges
from sift.domain.text.tokens import tokens
from sift.storage.database import Batches, Connection, Reading, reading, rows, run, writing
from sift.storage.keyword_lists import (
    blocks,
    dead_within,
    frequencies,
    keyword_generation,
    parts,
    positions,
    segments,
)
from sift.storage.keywords import totals
from tests.coded_lists import packed
from tests.keyword_stores import COLUMNS, Instance, Round, instances, texts

WORDS = ("alpha", "beta", "Gamma", "δέλτα", "x-ray", "naïve")


FIELDS = ("", "title: Front\n", "tags: [alpha, 2]\nscore: 1e20\nratio: 0.1\nok: true\n", "nested: {title: beta}\n")


text = st.lists(st.sampled_from(WORDS), max_size=40).map(" ".join)


content = st.tuples(st.sampled_from(FIELDS), text, text)


rounds = st.lists(
    st.dictionaries(st.sampled_from(["a.md", "b.md", "c.md", "d.md"]), st.one_of(st.none(), content), max_size=4),
    min_size=1,
    max_size=4,
)


class Term(NamedTuple):
    term: str


class Pack(NamedTuple):
    lists: bytes
    places: bytes


type Places = dict[str, dict[int, list[int]]]


def expected(found: list[Instance]) -> Places:
    sizes = Counter((each.chunk, each.column) for each in found)
    held: Places = {}
    for each in found:
        before = sum(sizes[each.chunk, column] + 1 for column in COLUMNS[: COLUMNS.index(each.column)])
        held.setdefault(each.term, {}).setdefault(each.chunk, []).append(before + each.offset)
    return {term: {chunk: sorted(places) for chunk, places in chunks.items()} for term, chunks in held.items()}


class ChunkPlace(NamedTuple):
    id: int
    place: int


def chunks_at(transaction: Reading) -> dict[int, int]:
    return {each.place: each.id for each in rows(transaction, ChunkPlace, "SELECT id, place FROM chunks")}


def stored(transaction: Reading) -> Places:
    chunk_at = chunks_at(transaction)
    gone = set(ints(dead_within(transaction, 0, 2**63 - 1)))
    packs = rows(transaction, Pack, "SELECT lists, places FROM keyword_packs")
    every = sorted({term for pack in packs for term, *_ in packed(pack.lists, pack.places)})
    held: Places = {}
    for part in parts(transaction, every):
        found = (
            [(part.data, positions(transaction, part.segment, part.term, None)[0].positions)]
            if not part.ranged
            else [
                (block.postings, place.positions)
                for block, place in zip(
                    blocks(transaction, part.segment, part.term, range(part.chunks)),
                    positions(transaction, part.segment, part.term, range(part.chunks)),
                    strict=True,
                )
            ]
        )
        for postings, places in found:
            ids, counts = blocked(postings)
            flat = positions_of(places, counts)
            for place, own in zip(ints(ids), np.split(flat, np.cumsum(counts)[:-1]), strict=True):
                if place not in gone:
                    held.setdefault(part.term, {})[chunk_at[place]] = own.tolist()
    return held


class Kept(NamedTuple):
    first: int
    last: int
    chunks: int
    dead: int


def checked(connection: Connection) -> None:
    with reading(connection) as transaction:
        found = instances(transaction)
        want = expected(found)
        assert stored(transaction) == want
        assert frequencies(transaction, sorted(want)) == {term: len(chunks) for term, chunks in want.items()}
        assert sorted(row.term for row in rows(transaction, Term, "SELECT term FROM keyword_terms")) == sorted(want)
        assert totals(transaction).tokens == len(found)
        held = segments(transaction)
        place_of = {chunk: place for place, chunk in chunks_at(transaction).items()}
        live = {place_of[chunk] for chunks in want.values() for chunk in chunks}
        assert all(sum(s.first <= place <= s.last for s in held) == 1 for place in live)
        gone = ints(dead_within(transaction, 0, 2**63 - 1))
        for each in rows(transaction, Kept, "SELECT first, last, chunks, dead FROM keyword_segments"):
            assert each.dead == sum(each.first <= chunk <= each.last for chunk in gone) < each.chunks


def generation_of(connection: Connection) -> int:
    with reading(connection) as transaction:
        return keyword_generation(transaction)


TWO = Batches(chunks_a_segment=2)


NOTES: Round = {f"n{n}.md": ("", f"Note {n}", "one two three " * n) for n in range(7)}


class Average(NamedTuple):
    first: int
    last: int
    chunks: int
    average: float


def chunk_tokens(transaction: Reading) -> dict[int, int]:
    return {
        chunk.chunk: len(tokens(chunk.title)) + len(tokens(chunk.text)) + len(tokens(chunk.meta or ""))
        for chunk in texts(transaction)
    }


def staled(connection: Connection) -> None:
    with writing(connection) as transaction:
        run(transaction, "UPDATE meta SET value = 'older' WHERE key = 'keyword_fingerprint'")
        run(transaction, "DELETE FROM keyword_packs")


MERGING = Batches(chunks_a_segment=2, documents_a_forget=1, postings_a_merge=3, merging=Policy(factor=2, floor=1))


class Sized(NamedTuple):
    chunks: int
    dead: int


def settled(connection: Connection, policy: Policy) -> list[Sized]:
    with reading(connection) as transaction:
        held = rows(transaction, Sized, "SELECT chunks, dead FROM keyword_segments ORDER BY first")
    assert merges([Held(each.chunks, each.dead) for each in held], policy) == ()
    return held


PAIRS = Batches(merging=Policy(factor=2, floor=1, span=0.0))


LONG: Round = {"a.md": ("", "a", "one two " * 5000)}
