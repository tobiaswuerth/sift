import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sift.domain.arrays import Ids, ints
from sift.domain.query.expressions import Or, Phrase, Term
from sift.retrieval.keyword.lists import (
    BlockKey,
    DecodedLists,
    Dense,
    PostingArrays,
    held_bytes,
    lengths_at,
    opened,
)
from sift.storage.database import Connection, reading
from tests.keyword_searches import Draws, prose, search
from tests.keyword_stores import SMALL, applied, opened_store

QUERY = Or(
    (Term("w0"), Term("w1"), Phrase(("w0", "w1")), Phrase(("w2", "w0")), Phrase(("w0", "w15")), Phrase(("w25", "w0")))
)
READS = ("keyword_lengths", "keyword_blocks", "lists, places FROM keyword_packs")


@contextmanager
def store_of(rounds: int, documents: int) -> Generator[Connection]:
    rng = Draws(np.random.default_rng(5))
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 2, SMALL) as connection:
        applied(connection, [prose(rng, documents, f"r{n}_") for n in range(rounds)])
        yield connection


def reads(said: list[str]) -> dict[str, int]:
    return {table: sum(table in statement for statement in said) for table in READS}


def alone(array: NDArray[np.generic]) -> bool:
    base = array.base
    return base is None or (base.base is None and base.nbytes == array.nbytes)


def test_a_search_given_what_was_decoded_before_reads_no_lengths_or_blocks_again() -> None:
    lengths, blocks, positions = dict[int, Dense](), dict[BlockKey, PostingArrays](), dict[BlockKey, Ids]()
    kept = DecodedLists(lengths, blocks, positions)
    with store_of(3, 30) as connection:
        said = list[str]()
        connection.set_trace_callback(said.append)
        with reading(connection) as transaction:
            first = search([opened(transaction, kept)], QUERY, 5)
        before, said[:] = reads(said), []
        with reading(connection) as transaction:
            again = search([opened(transaction, kept)], QUERY, 5)
        connection.set_trace_callback(None)
    assert (again, reads(said)) == (first, dict.fromkeys(READS, 0))
    assert all(before[table] for table in READS), before
    assert all(alone(each) for dense in lengths.values() for each in (dense.numbers, dense.table, dense.order))
    assert all(alone(each) for arrays in blocks.values() for each in arrays)
    assert all(alone(each) for each in positions.values())
    assert held_bytes(next(iter(positions.values()))) == next(iter(positions.values())).nbytes
    assert held_bytes(next(iter(blocks.values()))) == sum(each.nbytes for each in next(iter(blocks.values())))


def test_a_store_without_documents_searched_beside_another_changes_neither_its_reads_nor_its_hits() -> None:
    said = list[str]()
    with (
        store_of(3, 30) as connection,
        tempfile.TemporaryDirectory() as folder,
        opened_store(Path(folder), 3, SMALL) as empty,
    ):
        connection.set_trace_callback(said.append)
        with reading(connection) as transaction:
            alone_found = search([opened(transaction)], QUERY, 5)
        alone_read, said[:] = reads(said), []
        with reading(connection) as transaction, reading(empty) as nothing:
            beside = search([opened(transaction), opened(nothing)], QUERY, 5)
        connection.set_trace_callback(None)
    assert (beside, reads(said)) == (alone_found, alone_read)
    assert all(alone_read[table] for table in READS), alone_read


def test_lengths_given_to_a_search_stay_as_they_were_while_others_read_more_from_them() -> None:
    with store_of(1, 60) as connection, reading(connection) as transaction:
        base = DecodedLists({}, {}, {})
        store = opened(transaction, base)
        (segment,) = store.segments
        places = np.arange(segment.first, segment.last + 1, dtype=np.int64)
        block = places >> SMALL.lengths_shift
        third, fourth, fifth = ints(np.unique(block))[2:5]
        lengths_at(store, places[block <= third])
        shared = base.lengths.get(segment.id)
        assert shared is not None
        before = shared.table.copy()
        one, other = DecodedLists({segment.id: shared}, {}, {}), DecodedLists({segment.id: shared}, {}, {})
        read_by_one = lengths_at(opened(transaction, one), places[block == fourth])
        lengths_at(opened(transaction, other), places[block == fifth])
        assert np.array_equal(shared.table, before)
        assert np.array_equal(lengths_at(opened(transaction, one), places[block == fourth]), read_by_one)
        assert np.array_equal(read_by_one, lengths_at(opened(transaction), places[block == fourth]))
