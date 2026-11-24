import contextlib
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

from sift.api.ids import CollectionId
from sift.domain.arrays import Ids, ints
from sift.domain.codec.codes import Bits
from sift.domain.codec.order import topical
from sift.storage.database import Connection, Count, IdRow, reading, rows, run, run_each, writing
from sift.storage.keyword_order import order_due, ordered, placed_anew
from sift.storage.keywords import current, remade, remake_started, settle
from sift.storage.layout import LAYOUT
from sift.storage.store import create_store, open_store
from sift.storage.vectors import codes_of, forget_vectors, put_vectors
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


def embedded_pending(connection: Connection) -> None:
    with reading(connection) as transaction:
        pending = rows(transaction, IdRow, "SELECT chunk AS id FROM unembedded ORDER BY chunk")
    embedded(connection, [each.id for each in pending])


def written(connection: Connection) -> int:
    with reading(connection) as transaction:
        (found,) = rows(transaction, Count, "SELECT vectors_written AS n FROM counts")
    return found.n


def placed_in(connection: Connection, order: list[int]) -> int:
    embedded(connection, [each.id for each in placed(connection)])
    return placed_anew(connection, np.array(order, np.int64), written(connection))


def current_of(connection: Connection) -> bool:
    with reading(connection) as transaction:
        return current(transaction)


def due(connection: Connection) -> bool:
    with reading(connection) as transaction:
        return order_due(transaction)


TEN: Round = {f"m{n}.md": ("", f"M {n}", f"word{n} text") for n in range(10)}


def ordered_ten(connection: Connection) -> list[int]:
    applied(connection, [TEN])
    ids = [each.id for each in placed(connection)]
    embedded(connection, ids)
    assert ordered(connection) == len(ids)
    list(remade(connection))
    assert not due(connection)
    return ids


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
        assert placed_in(connection, [*ids[::-1][:-2], 10**15]) == len(ids)
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
        placed_in(connection, [each.id for each in placed(connection)][::-1])
        list(remade(connection))
        highest = max(each.place for each in placed(connection))
        applied(connection, [dict.fromkeys(NOTES), {"x.md": ("", "X", "four")}])
        (added,) = placed(connection)
        assert added.place == highest + 1
        checked(connection)


def test_a_chunk_given_its_place_moves_the_places_handed_out_past_it() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        highest = max(each.place for each in placed(connection))
        with writing(connection) as transaction:
            run(
                transaction,
                """INSERT INTO chunks(document, first_line, last_line, start, length, place)
                   SELECT document, 1, 1, 0, 1, ? FROM chunks LIMIT 1""",
                (highest + 100,),
            )
        applied(connection, [{"x.md": ("", "X", "four")}])
        assert placed(connection)[-1].place == highest + 101
        checked(connection)


def test_merges_and_changes_after_an_order_keep_every_chunk_at_its_one_place() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        placed_in(connection, [each.id for each in placed(connection)][::-1])
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


def test_a_store_embedded_anew_is_due_once_more_than_a_fifth_of_its_chunks_have_their_new_codes() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        ids = ordered_ten(connection)
        with writing(connection) as transaction:
            forget_vectors(transaction)
        assert not due(connection)
        embedded(connection, ids[:2])
        assert not due(connection)
        embedded(connection, ids[2:3])
        assert due(connection)


class Counted(NamedTuple):
    places: int
    vectors_written: int


def test_a_store_of_the_layout_before_places_places_its_chunks_by_id_and_is_due_once_for_its_vectors() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path, collection = Path(folder) / "3.sqlite", CollectionId(3)
        create_store(path, collection, LAYOUT[:7])
        with (
            contextlib.closing(open_store(path, collection, LAYOUT[:7])) as connection,
            writing(connection) as transaction,
        ):
            run(transaction, "INSERT INTO sources(id, path, modified_ns, size) VALUES (1, 'a.md', 0, 0)")
            run(
                transaction,
                "INSERT INTO documents(id, source, key, digest, title, fields) VALUES (1, 1, '', 'd', 'A', '{}')",
            )
            for _ in range(4):
                run(
                    transaction,
                    "INSERT INTO chunks(document, first_line, last_line, start, length) VALUES (1, 1, 1, 0, 1)",
                )
            ids = [each.id for each in rows(transaction, IdRow, "SELECT id FROM chunks ORDER BY id")]
            run(transaction, "DELETE FROM chunks WHERE id = ?", (ids[3],))
            run_each(transaction, "INSERT INTO vectors(chunk, vector) VALUES (?, x'00')", [(ids[0],), (ids[2],)])
        with contextlib.closing(open_store(path, collection)) as connection:
            assert placed(connection) == [Placed(each, each) for each in ids[:3]]
            with reading(connection) as transaction:
                assert rows(transaction, Counted, "SELECT places, vectors_written FROM counts") == [Counted(ids[3], 2)]
            assert due(connection)


def test_a_store_whose_documents_were_only_deleted_since_its_order_is_not_due() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        ordered_ten(connection)
        applied(connection, [dict.fromkeys(list(TEN)[:5])])
        assert not due(connection)


def test_a_store_is_due_an_order_once_more_than_a_fifth_of_its_chunks_got_codes_since_the_last() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = [each.id for each in placed(connection)]
        assert not due(connection)
        embedded(connection, ids[:1])
        assert not due(connection)
        embedded(connection, ids[1:2])
        assert due(connection)
        placed_anew(connection, np.array(ids[:2], np.int64), written(connection))
        list(remade(connection))
        assert not due(connection)
        embedded(connection, ids[2:3])
        assert not due(connection)
        embedded(connection, ids[3:4])
        assert due(connection)


def test_chunks_changed_since_an_order_count_once_toward_the_next_though_as_many_have_codes() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        ordered_ten(connection)
        applied(connection, [dict.fromkeys(list(TEN)[:2], ("", "Changed", "other words"))])
        embedded_pending(connection)
        assert not due(connection)
        applied(connection, [{list(TEN)[2]: ("", "Changed", "other words")}])
        embedded_pending(connection)
        assert due(connection)


def test_an_order_whose_codes_changed_a_little_while_it_was_made_is_placed_and_they_count_toward_the_next() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [TEN])
        ids = [each.id for each in placed(connection)]
        embedded(connection, ids)

        def racing(coded: Ids, bits: Bits) -> Ids:
            embedded(connection, ids[:1])
            return topical(coded, bits)

        assert ordered(connection, racing) == len(ids)
        list(remade(connection))
        assert not due(connection)
        embedded(connection, ids[1:3])
        assert due(connection)
        checked(connection)


def test_an_order_whose_codes_changed_enough_while_it_was_made_to_be_due_again_is_not_placed() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [TEN])
        ids = [each.id for each in placed(connection)]
        embedded(connection, ids)
        before = placed(connection)

        def racing(coded: Ids, bits: Bits) -> Ids:
            with writing(connection) as transaction:
                forget_vectors(transaction)
            embedded(connection, ids[:3])
            return topical(coded, bits)

        assert ordered(connection, racing) == 0
        assert (current_of(connection), placed(connection)) == (True, before)
        assert due(connection)


def test_an_order_read_before_another_was_placed_is_not_placed_once_the_store_is_no_longer_due() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [TEN])
        ids = [each.id for each in placed(connection)]
        embedded(connection, ids)
        with reading(connection) as transaction:
            first = topical(*codes_of(transaction))
        first_at = written(connection)
        embedded(connection, ids[:1])
        with reading(connection) as transaction:
            later = topical(*codes_of(transaction))
        later_at = written(connection)
        assert placed_anew(connection, first, first_at) == len(ids)
        assert placed_anew(connection, later[::-1], later_at) == 0
        list(remade(connection))
        assert placed_anew(connection, later[::-1], later_at) == 0
        assert [each.id for each in placed(connection)] == ints(first)
        checked(connection)


def test_an_order_made_from_codes_another_order_was_placed_by_is_not_placed_again() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        embedded(connection, [each.id for each in placed(connection)])
        with reading(connection) as transaction:
            ids, bits = codes_of(transaction)
        made_at = written(connection)
        assert placed_anew(connection, topical(ids, bits), made_at) == len(ids)
        list(remade(connection))
        after = placed(connection)
        assert placed_anew(connection, topical(ids, bits)[::-1], made_at) == 0
        assert (current_of(connection), placed(connection)) == (True, after)
        checked(connection)


def test_a_store_is_ordered_by_its_codes() -> None:
    asked: list[tuple[list[int], int]] = []

    def reversed_order(ids: Ids, bits: Bits) -> Ids:
        asked.append((ints(ids), bits.shape[1]))
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
        assert [each.id for each in placed(connection)] == ints(topical(np.array(ids, np.int64), bits)[::-1])
        assert not due(connection)
        checked(connection)


class Code(NamedTuple):
    chunk: int
    bits: bytes
