import json
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from sift.api.errors import StoreUnavailableError
from sift.domain.arrays import Ids, ints
from sift.domain.codec.codes import Bits, codes, width_of
from sift.domain.errors import DamagedError
from sift.storage.database import Reading, Writing, rows, run, run_each, streamed

type Halves = NDArray[np.float16]

MODEL_KEY, WIDTH_KEY = "vector_model", "vector_width"
PENDING = """SELECT c.id, d.title, substr(d.text, c.start + 1, c.length),
                    c.id = (SELECT min(f.id) FROM chunks f WHERE f.document = c.document)
             FROM chunks c JOIN documents d ON d.id = c.document
             WHERE c.id > ? AND NOT EXISTS (SELECT 1 FROM vectors v WHERE v.chunk = c.id)
             ORDER BY c.id LIMIT ?"""
EMBED_REMEDY = "run sift embed --reset to embed every collection with {model}"


class Pending(NamedTuple):
    chunk: int
    title: str
    text: str
    first: int


class Value(NamedTuple):
    value: str


class Coded(NamedTuple):
    chunk: int
    bits: bytes


class Stored(NamedTuple):
    chunk: int
    vector: bytes


class Count(NamedTuple):
    n: int


class Chunk(NamedTuple):
    id: int


def model_of(transaction: Reading) -> str | None:
    found = rows(transaction, Value, "SELECT value FROM meta WHERE key = ?", (MODEL_KEY,))
    return found[0].value if found else None


def pending(transaction: Reading, after: int, most: int) -> list[Pending]:
    return rows(transaction, Pending, PENDING, (after, most))


def left(transaction: Reading) -> int:
    (found,) = rows(transaction, Count, "SELECT max(chunks - vectors, 0) FROM counts")
    return found.n


def _held(transaction: Writing, model: str, width: int) -> None:
    found = model_of(transaction)
    if found not in {None, model}:
        raise StoreUnavailableError(
            f"the store holds vectors of {found}, not of {model}", EMBED_REMEDY.format(model=model)
        )
    run_each(
        transaction,
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        [(MODEL_KEY, model), (WIDTH_KEY, str(width))],
    )


def put_vectors(transaction: Writing, chunks: Ids, matrix: Halves, model: str) -> int:
    _held(transaction, model, width_of(matrix))
    there = {
        found.id
        for found in rows(
            transaction,
            Chunk,
            "SELECT id FROM chunks WHERE id IN (SELECT value FROM json_each(?))",
            (json.dumps(ints(chunks)),),
        )
    }
    intact = np.isfinite(matrix).all(axis=1)
    keep = intact & np.array([chunk in there for chunk in ints(chunks)], np.bool_)
    kept, rows_kept = ints(chunks[keep]), matrix[keep]
    run_each(
        transaction,
        "INSERT INTO vectors(chunk, vector) VALUES (?, ?) ON CONFLICT(chunk) DO UPDATE SET vector = excluded.vector",
        zip(kept, _rows(rows_kept), strict=True),
    )
    run_each(
        transaction,
        "INSERT OR REPLACE INTO codes(chunk, bits) VALUES (?, ?)",
        zip(kept, _rows(codes(rows_kept.astype(np.float32))), strict=True),
    )
    return len(kept)


def _rows(matrix: NDArray[np.generic]) -> list[bytes]:
    data = matrix.tobytes()
    size = len(data) // len(matrix) if len(matrix) else 0
    return [data[at : at + size] for at in range(0, len(data), size)] if size else []


def forget_vectors(transaction: Writing) -> None:
    run(transaction, "DELETE FROM vectors")
    run(transaction, "DELETE FROM meta WHERE key IN (?, ?)", (MODEL_KEY, WIDTH_KEY))


def codes_of(transaction: Reading) -> tuple[Ids, Bits]:
    found = list(streamed(transaction, Coded, "SELECT chunk, bits FROM codes ORDER BY chunk"))
    ids = np.fromiter((row.chunk for row in found), np.int64, len(found))
    width = len(found[0].bits) if found else 0
    bits = np.frombuffer(b"".join(row.bits for row in found), np.uint8).reshape(len(found), width)
    return ids, bits


def vectors_of(transaction: Reading, chunks: Ids) -> tuple[Ids, Halves]:
    found = rows(
        transaction,
        Stored,
        "SELECT chunk, vector FROM vectors WHERE chunk IN (SELECT value FROM json_each(?)) ORDER BY chunk",
        (json.dumps(ints(chunks)),),
    )
    widths = {len(row.vector) for row in found}
    if len(widths) > 1:
        raise DamagedError("the store's vectors are not all of one width")
    ids = np.fromiter((row.chunk for row in found), np.int64, len(found))
    matrix = np.frombuffer(b"".join(row.vector for row in found), np.float16)
    return ids, matrix.reshape(len(found), -1) if found else matrix.reshape(0, 0)
