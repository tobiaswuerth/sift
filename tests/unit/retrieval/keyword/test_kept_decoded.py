import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from sift.domain.codec.postings import HeldBlock, HeldTable, kept_bytes, postings_of
from sift.domain.query.expressions import Or, Phrase, Term
from sift.retrieval.keyword.lists import BlockKey, DecodedLists, opened
from sift.storage.database import Connection, reading
from tests.keyword_searches import Draws, prose, search
from tests.keyword_stores import SMALL, applied, opened_store

QUERY = Or(
    (Term("w0"), Term("w1"), Phrase(("w0", "w1")), Phrase(("w2", "w0")), Phrase(("w0", "w15")), Phrase(("w25", "w0")))
)
KEPT = ("keyword_lengths", "postings FROM keyword_blocks")
READS = (*KEPT, "positions FROM keyword_blocks", "lists, places FROM keyword_packs")


@contextmanager
def store_of(rounds: int, documents: int) -> Generator[Connection]:
    rng = Draws(np.random.default_rng(5))
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 2, SMALL) as connection:
        applied(connection, [prose(rng, documents, f"r{n}_") for n in range(rounds)])
        yield connection


def reads(said: list[str]) -> dict[str, int]:
    return {table: sum(table in statement for statement in said) for table in READS}


def test_a_search_given_what_was_decoded_before_reads_no_lengths_or_blocks_again() -> None:
    lengths, blocks = dict[int, HeldTable](), dict[BlockKey, HeldBlock]()
    kept = DecodedLists(lengths, blocks)
    with store_of(3, 30) as connection:
        said = list[str]()
        connection.set_trace_callback(said.append)
        with reading(connection) as transaction:
            first = search([opened(transaction, kept)], QUERY, 5)
        before, said[:] = reads(said), []
        with reading(connection) as transaction:
            again = search([opened(transaction, kept)], QUERY, 5)
        connection.set_trace_callback(None)
    assert (again, [reads(said)[table] > 0 for table in READS]) == (first, [table not in KEPT for table in READS])
    assert all(before[table] for table in READS), before
    assert {kept_bytes(each) % (4 << SMALL.lengths_shift) for each in lengths.values()} == {0}
    assert sum(kept_bytes(each) for each in lengths.values()) > 0
    assert all(kept_bytes(each) == sum(held.nbytes for held in postings_of([each])[:2]) for each in blocks.values())


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


def test_a_segment_s_lengths_are_one_table_that_the_searches_after_fill_further() -> None:
    lengths, blocks = dict[int, HeldTable](), dict[BlockKey, HeldBlock]()
    kept = DecodedLists(lengths, blocks)
    with store_of(1, 300) as connection:
        with reading(connection) as transaction:
            search([opened(transaction, kept)], Term("w20"), 5)
        (segment,) = lengths
        table, before = lengths[segment], kept_bytes(lengths[segment])
        with reading(connection) as transaction:
            search([opened(transaction, kept)], Or((Term("w0"), Term("w3"), Term("w7"))), 5)
    assert (list(lengths), kept_bytes(table) > before, kept_bytes(lengths[segment])) == (
        [segment],
        True,
        kept_bytes(table),
    )
