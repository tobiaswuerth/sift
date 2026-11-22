import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from sift.api.errors import StoreUnavailableError
from sift.domain.arrays import Ids, ints, last
from sift.domain.codec.codes import codes
from sift.domain.errors import DamagedError
from sift.storage.database import Connection, reading, rows, run, writing
from sift.storage.vectors import Halves, codes_of, forget_vectors, left, model_of, pending, put_vectors, vectors_of
from tests.keyword_stores import SMALL, Round, applied, opened_store

NOTES: Round = {"a.md": ("", "Alpha", "one two " * 2000), "b.md": ("", "Beta", "three")}


class Count(NamedTuple):
    vectors: int


def chunk_ids(connection: Connection) -> Ids:
    with reading(connection) as transaction:
        return np.array([row.chunk for row in pending(transaction, 0, 1000)], np.int64)


def vectors(n: int, width: int = 16, seed: int = 0) -> Halves:
    found = np.random.default_rng(seed).normal(size=(n, width))
    return (found / np.linalg.norm(found, axis=1, keepdims=True)).astype(np.float16)


def test_vectors_are_kept_with_their_codes_and_counted() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)
        matrix = vectors(len(ids))
        with writing(connection) as transaction:
            assert put_vectors(transaction, ids, matrix, "model-a") == len(ids)
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
            assert [(row.title, row.first) for row in waiting][:2] == [("Alpha", 1), ("Alpha", 0)]
            assert waiting[-1].title == "Beta"
            assert waiting[-1].first == 1
            assert pending(transaction, waiting[0].chunk, 1) == waiting[1:2]


def test_vectors_go_with_their_chunks_and_a_broken_or_gone_chunk_gets_none() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)
        matrix = vectors(len(ids))
        matrix[0, 3] = np.nan
        with writing(connection) as transaction:
            assert (
                put_vectors(transaction, np.append(ids, last(ids) + 99), np.vstack([matrix, matrix[:1]]), "m")
                == len(ids) - 1
            )
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
            assert put_vectors(transaction, ids, vectors(len(ids)), "model-b") == len(ids)


def test_vectors_of_more_than_one_width_are_damage() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        applied(connection, [NOTES])
        ids = chunk_ids(connection)
        with writing(connection) as transaction:
            put_vectors(transaction, ids, vectors(len(ids)), "m")
            run(transaction, "UPDATE vectors SET vector = x'0000' WHERE chunk = ?", (int(last(ids)),))
        with reading(connection) as transaction, pytest.raises(DamagedError, match="not all of one width"):
            vectors_of(transaction, ids)
