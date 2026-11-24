import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

from sift.domain.arrays import Ids
from sift.domain.codec.codes import Bits
from sift.domain.codec.order import topical
from sift.storage.database import Connection, reading, rows, writing
from sift.storage.keyword_order import order_due, ordered, placed_anew
from sift.storage.keywords import current, remade, remake_started, settle
from sift.storage.vectors import forget_vectors, put_vectors
from tests.keyword_indexes import MERGING, NOTES, TWO, checked
from tests.keyword_stores import SMALL, Round, applied, opened_store

WIDTH = 64


class Placed(NamedTuple):
    id: int
    place: int


def placed(connection: Connection) -> list[Placed]:
    with reading(connection) as transaction:
        return rows(transaction, Placed, "SELECT id, place FROM chunks ORDER BY place")


def embedded(connection: Connection, chunks: list[int]) -> None:
    matrix = np.random.default_rng(len(chunks)).standard_normal((len(chunks), WIDTH)).astype(np.float16)
    with writing(connection) as transaction:
        put_vectors(transaction, np.array(chunks, np.int64), matrix, "fake")


def test_a_chunk_takes_its_id_for_its_place_until_its_store_is_ordered() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES, {"n6.md": None}, {"x.md": ("", "X", "four five")}])
        held = placed(connection)
        assert len(held) == len(NOTES)
        assert all(each.place == each.id for each in held)


def test_an_index_placed_anew_holds_what_it_held_in_the_order_given_new_chunks_after() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = [each.id for each in placed(connection)]
        before = max(each.place for each in placed(connection))
        order = [*ids[::-1][:-2], 10**15]
        assert placed_anew(connection, np.array(order, np.int64)) == len(ids)
        assert not current_of(connection)
        assert remake_started(connection) == len(ids)
        assert list(remade(connection, TWO))[-1] == len(ids)
        assert current_of(connection)
        held = placed(connection)
        assert [each.id for each in held] == [*ids[::-1][:-2], *ids[:2]]
        assert [each.place for each in held] == list(range(before + 1, before + 1 + len(ids)))
        checked(connection)
        applied(connection, [{"x.md": ("", "X", "four")}])
        assert placed(connection)[-1].place == before + len(ids) + 1
        checked(connection)


def test_a_chunk_added_once_the_highest_placed_are_gone_takes_a_place_never_held_and_is_indexed() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = [each.id for each in placed(connection)]
        placed_anew(connection, np.array(ids[::-1], np.int64))
        list(remade(connection))
        highest = max(each.place for each in placed(connection))
        applied(connection, [dict.fromkeys(NOTES), {"x.md": ("", "X", "four")}])
        (added,) = placed(connection)
        assert added.place == highest + 1
        checked(connection)


def test_merges_and_changes_after_an_order_keep_every_chunk_at_its_one_place() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = [each.id for each in placed(connection)]
        placed_anew(connection, np.array(ids[::-1], np.int64))
        list(remade(connection, TWO))
        before = {each.id: each.place for each in placed(connection)}
        rounds: list[Round] = [{f"n{n}.md": ("", f"Changed {n}", "four " * n)} for n in range(0, 7, 2)]
        applied(connection, rounds, MERGING)
        list(settle(connection, MERGING))
        after = {each.id: each.place for each in placed(connection)}
        assert {chunk: place for chunk, place in after.items() if chunk in before} == {
            chunk: place for chunk, place in before.items() if chunk in after
        }
        assert len(set(after.values())) == len(after)
        checked(connection)


def test_a_store_embedded_anew_is_ordered_anew() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = [each.id for each in placed(connection)]
        embedded(connection, ids)
        ordered(connection)
        list(remade(connection))
        assert not due(connection)
        with writing(connection) as transaction:
            forget_vectors(transaction)
        assert not due(connection)
        embedded(connection, ids)
        assert due(connection)


def current_of(connection: Connection) -> bool:
    with reading(connection) as transaction:
        return current(transaction)


def due(connection: Connection) -> bool:
    with reading(connection) as transaction:
        return order_due(transaction)


def test_a_store_is_due_an_order_once_more_than_a_fifth_of_its_chunks_got_codes_since_the_last() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = [each.id for each in placed(connection)]
        assert not due(connection)
        embedded(connection, ids[:1])
        assert not due(connection)
        embedded(connection, ids[1:2])
        assert due(connection)
        placed_anew(connection, np.array(ids[:2], np.int64))
        list(remade(connection))
        assert not due(connection)
        embedded(connection, ids[2:3])
        assert not due(connection)
        embedded(connection, ids[3:4])
        assert due(connection)


def test_a_store_is_ordered_by_its_codes() -> None:
    asked: list[tuple[list[int], int]] = []

    def reversed_order(ids: Ids, bits: Bits) -> Ids:
        asked.append((ids.tolist(), bits.shape[1]))
        return topical(ids, bits)[::-1]

    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = [each.id for each in placed(connection)]
        embedded(connection, ids)
        assert ordered(connection, reversed_order) == len(ids)
        list(remade(connection))
        with reading(connection) as transaction:
            codes = rows(transaction, Code, "SELECT chunk, bits FROM codes ORDER BY chunk")
        bits = np.stack([np.frombuffer(each.bits, np.uint8) for each in codes])
        assert asked == [(ids, bits.shape[1])]
        assert [each.id for each in placed(connection)] == topical(np.array(ids, np.int64), bits)[::-1].tolist()
        assert not due(connection)
        checked(connection)


class Code(NamedTuple):
    chunk: int
    bits: bytes
