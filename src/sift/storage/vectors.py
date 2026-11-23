import json
from typing import NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from sift.api.errors import StoreUnavailableError
from sift.domain.arrays import STORED_HALF, Ids, among, ascending, distinct, ints
from sift.domain.codec.codes import MOST_BITS, Bits, Halves, code_bytes, codes, width_of
from sift.domain.errors import DamagedError
from sift.storage.database import Reading, Writing, first, rows, run, run_each

MODEL_KEY, WIDTH_KEY = "vector_model", "vector_width"
PENDING = """SELECT c.id, c.document, c.start, c.length,
                    c.id = (SELECT min(f.id) FROM chunks f WHERE f.document = c.document)
             FROM unembedded u JOIN chunks c ON c.id = u.chunk
             WHERE u.chunk > ?
             ORDER BY u.chunk LIMIT ?"""
TEXTS = "SELECT id, title, text FROM documents WHERE id IN (SELECT value FROM json_each(?))"
CODES = "SELECT chunk, bits FROM codes ORDER BY chunk"
HELD_META = "SELECT value FROM meta WHERE key = ? AND (SELECT vectors FROM counts) > 0"
READ = 65_536
MATRIX = 2
EMBED_REMEDY = "run sift embed --reset to embed every collection with {model}"
DAMAGED_CODES = "the store's codes do not match its vectors"


class Pending(NamedTuple):
    chunk: int
    title: str
    text: str
    first: bool


class Kept(NamedTuple):
    stored: int
    broken: tuple[int, ...]


class Waiting(NamedTuple):
    chunk: int
    document: int
    start: int
    length: int
    first: int


class Text(NamedTuple):
    id: int
    title: str
    text: str


class Value(NamedTuple):
    value: str


class VectorRow(NamedTuple):
    chunk: int
    vector: bytes


class Count(NamedTuple):
    n: int


class Chunk(NamedTuple):
    id: int


def _meta(transaction: Reading, key: str) -> str | None:
    found = first(transaction, Value, HELD_META, (key,))
    return None if found is None else found.value


def model_of(transaction: Reading) -> str | None:
    return _meta(transaction, MODEL_KEY)


def width_held(transaction: Reading) -> int | None:
    found = _meta(transaction, WIDTH_KEY)
    return None if found is None else int(found)


def changes(transaction: Reading) -> int:
    (found,) = rows(transaction, Count, "SELECT vector_changes FROM counts")
    return found.n


def pending(transaction: Reading, after: int, most: int) -> list[Pending]:
    waiting = rows(transaction, Waiting, PENDING, (after, most))
    wanted = json.dumps(sorted({each.document for each in waiting}))
    texts = {held.id: held for held in rows(transaction, Text, TEXTS, (wanted,))}
    return [
        Pending(
            each.chunk,
            texts[each.document].title,
            texts[each.document].text[each.start : each.start + each.length],
            bool(each.first),
        )
        for each in waiting
    ]


def documents_embedded(transaction: Reading) -> int:
    sql = "SELECT count(DISTINCT c.document) FROM codes k JOIN chunks c ON c.id = k.chunk"
    (found,) = rows(transaction, Count, sql)
    return found.n


def left(transaction: Reading) -> int:
    (found,) = rows(transaction, Count, "SELECT chunks - vectors FROM counts")
    return found.n


def _checked(chunks: Ids, matrix: Halves) -> None:
    if matrix.ndim != MATRIX or matrix.dtype != np.float16:
        raise ValueError(f"vectors are kept as a 2-D float16 matrix, not {matrix.ndim}-D {matrix.dtype}: convert")
    if len(chunks) != len(matrix):
        raise ValueError(f"{len(chunks):,} chunks were given {len(matrix):,} vectors: give one vector a chunk")
    if not 0 < width_of(matrix) <= MOST_BITS:
        raise ValueError(f"vectors of {width_of(matrix):,} dimensions cannot be coded: from 1 to {MOST_BITS:,} can")
    if len(distinct(ascending(chunks)).values) != len(chunks):
        raise ValueError("a chunk was given more than one vector: give each chunk one")


def refuse_model(transaction: Reading, model: str) -> None:
    found = model_of(transaction)
    if found not in {None, model}:
        raise StoreUnavailableError(
            f"the store holds vectors of {found}, not of {model}", EMBED_REMEDY.format(model=model)
        )


def _refuse_another(transaction: Writing, model: str, width: int) -> None:
    refuse_model(transaction, model)
    held = width_held(transaction)
    if held not in {None, width}:
        raise StoreUnavailableError(
            f"the store holds vectors of {held:,} dimensions, not of {width:,}", EMBED_REMEDY.format(model=model)
        )


def put_vectors(transaction: Writing, chunks: Ids, matrix: Halves, model: str) -> Kept:
    _checked(chunks, matrix)
    _refuse_another(transaction, model, width_of(matrix))
    wanted = json.dumps(ints(chunks))
    sql = "SELECT id FROM chunks WHERE id IN (SELECT value FROM json_each(?))"
    there = among(
        chunks, ascending(np.array([found.id for found in rows(transaction, Chunk, sql, (wanted,))], np.int64))
    )
    intact = np.isfinite(matrix).all(axis=1)
    kept, rows_kept = ints(chunks[there & intact]), matrix[there & intact]
    if kept:
        run_each(
            transaction,
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            [(MODEL_KEY, model), (WIDTH_KEY, str(width_of(matrix)))],
        )
    run_each(
        transaction,
        "INSERT INTO vectors(chunk, vector) VALUES (?, ?) ON CONFLICT(chunk) DO UPDATE SET vector = excluded.vector",
        zip(kept, _rows(rows_kept.astype(STORED_HALF)), strict=True),
    )
    run_each(
        transaction,
        "INSERT OR REPLACE INTO codes(chunk, bits) VALUES (?, ?)",
        zip(kept, _rows(codes(rows_kept.astype(np.float32))), strict=True),
    )
    return Kept(len(kept), tuple(ints(chunks[there & ~intact])))


def _rows(matrix: NDArray[np.generic]) -> list[bytes]:
    data = matrix.tobytes()
    size = len(data) // len(matrix) if len(matrix) else 0
    return [data[at : at + size] for at in range(0, len(data), size)] if size else []


def forget_vectors(transaction: Writing) -> None:
    run(transaction, "DELETE FROM vectors")
    run(transaction, "DELETE FROM meta WHERE key IN (?, ?)", (MODEL_KEY, WIDTH_KEY))


def codes_of(transaction: Reading, read: int = READ) -> tuple[Ids, Bits]:
    (held,) = rows(transaction, Count, "SELECT vectors FROM counts")
    size = code_bytes(width_held(transaction) or 0)
    ids, bits = np.empty(held.n, np.int64), np.empty((held.n, size), np.uint8)
    done = 0
    cursor = transaction.connection.execute(CODES)
    while batch := cast("list[tuple[int, bytes]]", cursor.fetchmany(read)):
        if done + len(batch) > held.n or any(len(row[1]) != size for row in batch):
            raise DamagedError(DAMAGED_CODES)
        ids[done : done + len(batch)] = np.fromiter((row[0] for row in batch), np.int64, len(batch))
        bits[done : done + len(batch)] = np.frombuffer(b"".join(row[1] for row in batch), np.uint8).reshape(-1, size)
        done += len(batch)
    if done != held.n:
        raise DamagedError(DAMAGED_CODES)
    return ids, bits


def vectors_of(transaction: Reading, chunks: Ids) -> tuple[Ids, Halves]:
    found = rows(
        transaction,
        VectorRow,
        "SELECT chunk, vector FROM vectors WHERE chunk IN (SELECT value FROM json_each(?)) ORDER BY chunk",
        (json.dumps(ints(chunks)),),
    )
    width = width_held(transaction) or 0
    if any(len(row.vector) != 2 * width for row in found):
        raise DamagedError(f"a vector of the store is not of its {width:,} dimensions")
    ids = np.fromiter((row.chunk for row in found), np.int64, len(found))
    stored = np.frombuffer(b"".join(row.vector for row in found), STORED_HALF)
    return ids, stored.astype(np.float16).reshape(len(found), width)
