import contextlib
import tempfile
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from sift.api.ids import CollectionId
from sift.domain.arrays import ints
from sift.domain.codec.postings import blocked, positions_of
from sift.domain.codec.segments import Shape
from sift.storage.database import Batches, Connection, Reading, WritingDocuments, reading, rows, run, writing
from sift.storage.keyword_lists import blocks, dead, expansions, frequencies, parts, positions, segments
from sift.storage.keywords import current, totals
from sift.storage.store import LAYOUT, create_store, open_store
from tests.keyword_stores import COLUMNS, SMALL, Instance, Round, applied, changed, instances, opened_store

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


type Places = dict[str, dict[int, list[int]]]


def expected(found: list[Instance]) -> Places:
    sizes = Counter((each.chunk, each.column) for each in found)
    held: Places = {}
    for each in found:
        before = sum(sizes[each.chunk, column] + 1 for column in COLUMNS[: COLUMNS.index(each.column)])
        held.setdefault(each.term, {}).setdefault(each.chunk, []).append(before + each.offset)
    return {term: {chunk: sorted(places) for chunk, places in chunks.items()} for term, chunks in held.items()}


def stored(transaction: Reading) -> Places:
    gone = set(ints(dead(transaction)))
    every = sorted({row.term for row in rows(transaction, Term, "SELECT DISTINCT term FROM keyword_postings")})
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
            for chunk, own in zip(ints(ids), np.split(flat, np.cumsum(counts)[:-1]), strict=True):
                if chunk not in gone:
                    held.setdefault(part.term, {})[chunk] = own.tolist()
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
        live = {chunk for chunks in want.values() for chunk in chunks}
        assert all(sum(s.first <= chunk <= s.last for s in held) == 1 for chunk in live)
        gone = ints(dead(transaction))
        for each in rows(transaction, Kept, "SELECT first, last, chunks, dead FROM keyword_segments"):
            assert each.dead == sum(each.first <= chunk <= each.last for chunk in gone) < each.chunks


@settings(max_examples=40, deadline=None)
@given(rounds, st.sampled_from([SMALL, Shape()]))
def test_the_keyword_index_holds_what_fts5_makes_of_the_live_chunks_however_they_changed(
    each: list[Round], shape: Shape
) -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, shape) as connection:
        applied(connection, each)
        checked(connection)


@settings(max_examples=20, deadline=None)
@given(rounds)
def test_segments_of_few_chunks_and_documents_forgotten_a_few_at_a_time_index_the_same(each: list[Round]) -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, each, Batches(chunks_a_segment=2, documents_a_forget=1))
        checked(connection)


def test_an_index_of_another_version_is_made_anew_by_the_next_write() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [{"a.md": ("", "a", "one two"), "b.md": ("", "b", "three one")}])
        with writing(connection) as transaction:
            run(transaction, "UPDATE meta SET value = 'older' WHERE key = 'keyword_fingerprint'")
            run(transaction, "DELETE FROM keyword_postings")
        with reading(connection) as transaction:
            assert not current(transaction)
        applied(connection, [{}])
        with reading(connection) as transaction:
            assert current(transaction)
        checked(connection)


def test_a_store_of_the_layout_before_the_keyword_index_gets_one_by_the_next_write() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "3.sqlite"
        create_store(path, CollectionId(3), LAYOUT[:1])
        with (
            contextlib.closing(open_store(path, CollectionId(3), LAYOUT[:1])) as connection,
            writing(connection) as transaction,
        ):
            run(transaction, "DELETE FROM meta WHERE key LIKE 'keyword%'")
            changed(WritingDocuments(connection), "a.md", ("tags: [old]\n", "Old", "kept from before"))
        with contextlib.closing(open_store(path, CollectionId(3))) as connection:
            with reading(connection) as transaction:
                assert not current(transaction)
            applied(connection, [{"b.md": ("", "New", "added after")}])
            checked(connection)
            with reading(connection) as transaction:
                assert frequencies(transaction, ["old", "added"]) == {"old": 1, "added": 1}


def test_a_prefix_expands_to_the_terms_it_begins() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [{"a.md": ("", "x", "super superb superconduct supra s")}])
        with reading(connection) as transaction:
            assert expansions(transaction, "super") == ["super", "superb", "superconduct"]
            assert expansions(transaction, "s") == ["s", "super", "superb", "superconduct", "supra"]
            assert expansions(transaction, "zz") == []
            assert expansions(transaction, chr(0x10FFFF)) == []
            assert expansions(transaction, "a" + chr(0x10FFFF)) == []
            assert expansions(transaction, chr(0xD7FF)) == []


def test_a_segment_of_chunks_all_gone_goes_with_them() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [{"a.md": ("", "a", "one two")}, {"b.md": ("", "b", "three")}, {"a.md": None}])
        with reading(connection) as transaction:
            (left,) = segments(transaction)
            assert dead(transaction).tolist() == []
            assert frequencies(transaction, ["one", "three"]) == {"three": 1}
            assert left.chunks == 1
