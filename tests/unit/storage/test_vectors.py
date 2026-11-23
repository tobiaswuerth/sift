import contextlib
import tempfile
import tracemalloc
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from sift.api.errors import StoreUnavailableError
from sift.api.ids import CollectionId
from sift.domain.arrays import Ids, first, ints, last
from sift.domain.codec.codes import Halves, codes
from sift.domain.errors import DamagedError
from sift.storage.database import Connection, Reading, reading, rows, run, writing
from sift.storage.store import LAYOUT, open_store
from sift.storage.vectors import (
    VectorsPut,
    codes_of,
    forget_vectors,
    left,
    model_of,
    pending,
    put_vectors,
    vectors_of,
    width_held,
)
from tests.keyword_stores import SMALL, Round, applied, opened_store

NOTES: Round = {"a.md": ("", "Alpha", "one two " * 2000), "b.md": ("", "Beta", "three")}


class Count(NamedTuple):
    vectors: int


def chunk_ids(connection: Connection, most: int = 1000) -> Ids:
    with reading(connection) as transaction:
        return np.array([row.chunk for row in pending(transaction, 0, most)], np.int64)


def vectors(n: int, width: int = 16, seed: int = 0) -> Halves:
    found = np.random.default_rng(seed).normal(size=(n, width))
    return (found / np.linalg.norm(found, axis=1, keepdims=True)).astype(np.float16)


def test_vectors_are_kept_with_their_codes_and_counted() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)
        matrix = vectors(len(ids))
        with writing(connection) as transaction:
            assert put_vectors(transaction, ids, matrix, "model-a") == VectorsPut(len(ids), ())
        with reading(connection) as transaction:
            found, held = vectors_of(transaction, ids[::-1])
            assert ints(found) == ints(ids)
            assert bool(np.array_equal(held, matrix))
            coded, bits = codes_of(transaction)
            assert ints(coded) == ints(ids)
            assert bool(np.array_equal(bits, codes(matrix.astype(np.float32))))
            assert model_of(transaction) == "model-a"
            assert left(transaction) == 0
            assert pending(transaction, 0, 10) == []


def test_chunks_without_a_vector_are_pending_with_their_title_and_whether_they_come_first() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        with reading(connection) as transaction:
            waiting = pending(transaction, 0, 1000)
            assert left(transaction) == len(waiting) > 2
            assert [(row.title, row.first) for row in waiting][:2] == [("Alpha", True), ("Alpha", False)]
            assert waiting[-1].title == "Beta"
            assert waiting[-1].first is True
            assert pending(transaction, waiting[0].chunk, 1) == waiting[1:2]


def test_chunks_waiting_for_vectors_are_found_without_looking_at_those_embedded() -> None:
    many: Round = {f"{n:04}.md": ("", "T", f"w{n}") for n in range(2000)}
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [many])
        ids = chunk_ids(connection, 5000)
        with writing(connection) as transaction:
            put_vectors(transaction, ids, vectors(len(ids)), "m")
        applied(connection, [{"new.md": ("", "New", "fresh words")}])
        ticks: list[int] = []

        def tick() -> int:
            ticks.append(1)
            return 0

        connection.set_progress_handler(tick, 100)
        with reading(connection) as transaction:
            (waiting,) = pending(transaction, 0, 1024)
        connection.set_progress_handler(None, 0)
        assert waiting.title == "New"
        assert len(ticks) < 20


def test_chunks_wait_for_vectors_again_once_forgotten_and_not_once_gone() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)
        with writing(connection) as transaction:
            put_vectors(transaction, ids, vectors(len(ids)), "m")
        applied(connection, [{"a.md": None}])
        with reading(connection) as transaction:
            assert pending(transaction, 0, 1000) == []
        with writing(connection) as transaction:
            forget_vectors(transaction)
        with reading(connection) as transaction:
            assert [row.title for row in pending(transaction, 0, 1000)] == ["Beta"]


def test_a_store_of_the_layout_before_lists_the_chunks_without_vectors_once_opened() -> None:
    with tempfile.TemporaryDirectory() as folder:
        with opened_store(Path(folder), 3, SMALL) as connection:
            applied(connection, [NOTES])
            ids = chunk_ids(connection)
            with writing(connection) as transaction:
                put_vectors(transaction, ids[::2], vectors(len(ids[::2])), "m")
                for trigger in ("chunks_unembedded", "chunks_gone_unembedded", "vectors_embedded", "vectors_forgotten"):
                    run(transaction, f"DROP TRIGGER {trigger}")
                run(transaction, "DROP TABLE unembedded")
                run(transaction, "ALTER TABLE keyword_segments ADD COLUMN tokens INTEGER NOT NULL DEFAULT 0")
                run(transaction, f"PRAGMA user_version = {len(LAYOUT) - 2}")
        with contextlib.closing(open_store(Path(folder) / "3.sqlite", CollectionId(3))) as connection:
            assert ints(chunk_ids(connection)) == ints(ids[1::2])


def test_vectors_go_with_their_chunks_and_a_broken_or_gone_chunk_gets_none() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)
        matrix = vectors(len(ids))
        matrix[0, 3] = np.nan
        with writing(connection) as transaction:
            kept = put_vectors(transaction, np.append(ids, last(ids) + 99), np.vstack([matrix, matrix[:1]]), "m")
            assert kept == VectorsPut(len(ids) - 1, (first(ids),))
        applied(connection, [{"b.md": None}])
        with reading(connection) as transaction:
            (counted,) = rows(transaction, Count, "SELECT vectors FROM counts")
            coded, _ = codes_of(transaction)
            assert counted.vectors == len(coded) == len(ids) - 2
            assert [row.chunk for row in pending(transaction, 0, 10)] == ints(ids[:1])


def test_vectors_of_another_model_are_refused_until_forgotten() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)
        with writing(connection) as transaction:
            put_vectors(transaction, ids, vectors(len(ids)), "model-a")
        refusal = "holds vectors of model-a, not of model-b"
        with pytest.raises(StoreUnavailableError, match=refusal) as refused, writing(connection) as transaction:
            put_vectors(transaction, ids, vectors(len(ids)), "model-b")
        assert refused.value.remedy == "run sift embed --reset to embed every collection with model-b"
        with writing(connection) as transaction:
            forget_vectors(transaction)
        with reading(connection) as transaction:
            assert (model_of(transaction), left(transaction), len(codes_of(transaction)[0])) == (None, len(ids), 0)
        with writing(connection) as transaction:
            assert put_vectors(transaction, ids, vectors(len(ids)), "model-b").stored == len(ids)


def test_a_vector_not_of_the_stores_width_is_damage() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)
        with writing(connection) as transaction:
            put_vectors(transaction, ids, vectors(len(ids)), "m")
            run(transaction, "UPDATE vectors SET vector = x'0000' WHERE chunk = ?", (int(last(ids)),))
        with reading(connection) as transaction, pytest.raises(DamagedError, match="not of its 16 dimensions"):
            vectors_of(transaction, ids)


class Changes(NamedTuple):
    vector_changes: int


def changes(transaction: Reading) -> int:
    (found,) = rows(transaction, Changes, "SELECT vector_changes FROM counts")
    return found.vector_changes


def test_a_vector_put_again_replaces_its_code_and_marks_a_change() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)
        with writing(connection) as transaction:
            put_vectors(transaction, ids, vectors(len(ids)), "m")
        with reading(connection) as transaction:
            before = changes(transaction)
        again = -vectors(len(ids))
        with writing(connection) as transaction:
            put_vectors(transaction, ids[:1], again[:1], "m")
        with reading(connection) as transaction:
            assert bool(np.array_equal(vectors_of(transaction, ids[:1])[1], again[:1]))
            assert bool(np.array_equal(codes_of(transaction)[1][:1], codes(again[:1].astype(np.float32))))
            assert (left(transaction), changes(transaction)) == (0, before + 1)
        applied(connection, [{"b.md": None}])
        with reading(connection) as transaction:
            assert changes(transaction) == before + 2


def test_vectors_of_another_width_under_the_same_model_are_refused() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)
        with writing(connection) as transaction:
            put_vectors(transaction, ids, vectors(len(ids)), "m")
        with (
            pytest.raises(StoreUnavailableError, match="vectors of 16 dimensions, not of 32") as refused,
            writing(connection) as transaction,
        ):
            put_vectors(transaction, ids, vectors(len(ids), 32), "m")
        assert refused.value.remedy == "run sift embed --reset to embed every collection with m"


@pytest.mark.parametrize(
    "matrix, refusal",
    [
        (np.zeros((2, 16), np.float32), "2-D float16 matrix, not 2-D float32"),
        (np.zeros(16, np.float16), "2-D float16 matrix, not 1-D float16"),
        (np.zeros((3, 16), np.float16), "2 chunks were given 3 vectors"),
        (np.zeros((2, 65_536), np.float16), "vectors of 65,536 dimensions cannot be coded"),
        (np.zeros((2, 0), np.float16), "vectors of 0 dimensions cannot be coded"),
    ],
)
def test_vectors_that_cannot_be_kept_as_given_are_refused(matrix: Halves, refusal: str) -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)[:2]
        with writing(connection) as transaction, pytest.raises(ValueError, match=refusal):
            put_vectors(transaction, ids, matrix, "m")


def test_a_store_never_embedded_holds_no_codes_and_no_width() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        with reading(connection) as transaction:
            ids, bits = codes_of(transaction)
            assert (ids.shape, bits.shape, width_held(transaction)) == ((0,), (0, 0), None)


@pytest.mark.parametrize(
    "damage",
    [
        "DELETE FROM codes WHERE chunk = (SELECT max(chunk) FROM codes)",
        "UPDATE codes SET bits = x'00' WHERE chunk = (SELECT max(chunk) FROM codes)",
        "UPDATE counts SET vectors = vectors - 1",
        "UPDATE counts SET vectors = vectors - 2",
    ],
)
def test_codes_that_do_not_match_the_vectors_are_damage(damage: str) -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)
        with writing(connection) as transaction:
            put_vectors(transaction, ids, vectors(len(ids)), "m")
            run(transaction, damage)
        with reading(connection) as transaction, pytest.raises(StoreUnavailableError, match="codes do not match"):
            codes_of(transaction, read=2)


def test_codes_are_read_in_batches_into_arrays_made_once() -> None:
    notes: Round = {f"{at}.md": ("", "T", f"w{at}") for at in range(1500)}
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [notes])
        ids = chunk_ids(connection, 2000)
        with writing(connection) as transaction:
            put_vectors(transaction, ids, vectors(len(ids), 2048), "m")
        with reading(connection) as transaction:
            codes_of(transaction, read=100)
            tracemalloc.start()
            found, bits = codes_of(transaction, read=100)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        assert ints(found) == ints(ids)
        assert peak < 1.5 * (found.nbytes + bits.nbytes)


def test_a_store_given_only_broken_vectors_holds_no_model() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)
        with writing(connection) as transaction:
            assert put_vectors(transaction, ids[:1], np.full((1, 16), np.nan, np.float16), "m") == VectorsPut(
                0, (first(ids),)
            )
        with reading(connection) as transaction:
            assert (model_of(transaction), width_held(transaction)) == (None, None)


class Cut(NamedTuple):
    text: str


CUT = """SELECT substr(d.text, c.start + 1, c.length)
         FROM chunks c JOIN documents d ON d.id = c.document ORDER BY c.id"""


def test_pending_chunks_hold_their_text_cut_by_characters() -> None:
    notes: Round = {"a.md": ("", "Alpha", "thrée 😀 x " * 3000), "b.md": ("", "Beta", "🎉 two")}
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [notes])
        with reading(connection) as transaction:
            cut = rows(transaction, Cut, CUT)
            waiting = pending(transaction, 0, 1000)
        assert len(waiting) > 2
        assert [row.text for row in waiting] == [row.text for row in cut]


def test_a_store_whose_vectors_all_went_with_their_documents_takes_another_model() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)
        with writing(connection) as transaction:
            put_vectors(transaction, ids, vectors(len(ids)), "model-a")
        applied(connection, [{"a.md": None, "b.md": None}, {"c.md": ("", "Gamma", "four")}])
        with reading(connection) as transaction:
            assert (model_of(transaction), width_held(transaction)) == (None, None)
        ids = chunk_ids(connection)
        with writing(connection) as transaction:
            assert put_vectors(transaction, ids, vectors(len(ids), 32), "model-b").stored == len(ids)
        with reading(connection) as transaction:
            assert (model_of(transaction), width_held(transaction)) == ("model-b", 32)


def test_a_chunk_given_two_vectors_is_refused() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)
        with writing(connection) as transaction, pytest.raises(ValueError, match="more than one vector"):
            put_vectors(transaction, np.append(ids, ids[:1]), vectors(len(ids) + 1), "m")
