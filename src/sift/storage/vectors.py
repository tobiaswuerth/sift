import json
from collections.abc import Iterator
from typing import NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from sift.api.errors import Problem, StoreUnavailableError
from sift.api.ids import ChunkId, DocumentId
from sift.domain.arrays import STORED_HALF, Ids, among, ascending, distinct, ints
from sift.domain.codec.codes import (
    MOST_BITS,
    Bits,
    Halves,
    HeldCodes,
    code_bytes,
    codes,
    filled,
    filling,
    finished,
    width_of,
)
from sift.domain.errors import DamagedError
from sift.domain.text.embedding import RECIPE
from sift.storage.database import Count, IdRow, Reading, Writing, rows, run, run_each, touched
from sift.storage.texts import TextSpan, spans_text
from sift.storage.vector_meta import MODEL_KEY, RECIPE_KEY, WIDTH_KEY, model_of, recipe_of, width_held

PENDING = """SELECT c.id, c.document, c.start, c.length,
                    c.id = (SELECT min(f.id) FROM chunks f WHERE f.document = c.document)
             FROM unembedded u JOIN chunks c ON c.id = u.chunk
             WHERE u.chunk > ?
             ORDER BY u.chunk LIMIT ?"""
TITLES = "SELECT id, title FROM documents WHERE id IN (SELECT value FROM json_each(?))"
CODES = "SELECT chunk, bits FROM codes ORDER BY chunk"
VECTORS = "SELECT chunk, vector FROM vectors WHERE chunk IN (SELECT value FROM json_each(?)) ORDER BY chunk"
TOUCH_VECTORS = "SELECT sum(length(vector)) FROM vectors WHERE chunk IN (SELECT value FROM json_each(?))"
READ = 65_536
MATRIX = 2
EMBED_REMEDY = "embed again with reset, to embed every collection with {model}"
DAMAGED_CODES = "the store's codes do not match its vectors"
CODES_REMEDY = "embed again with reset, which embeds every collection again"


class Pending(NamedTuple):
    chunk: ChunkId
    title: str
    text: str
    first: bool


class VectorsPut(NamedTuple):
    stored: int
    broken: tuple[int, ...]


class Waiting(NamedTuple):
    chunk: ChunkId
    document: DocumentId
    start: int
    length: int
    first: int


class TitleRow(NamedTuple):
    id: int
    title: str


class VectorRow(NamedTuple):
    chunk: ChunkId
    vector: bytes


def pending(transaction: Reading, after: int, most: int) -> list[Pending]:
    spans = rows(transaction, Waiting, PENDING, (after, most))
    wanted = json.dumps(sorted({each.document for each in spans}))
    titles = {held.id: held.title for held in rows(transaction, TitleRow, TITLES, (wanted,))}
    cut = spans_text(transaction, [TextSpan(each.document, each.start, each.length) for each in spans])
    return [
        Pending(each.chunk, titles[each.document], text, bool(each.first))
        for each, text in zip(spans, cut, strict=True)
    ]


def documents_embedded(transaction: Reading) -> int:
    sql = "SELECT count(DISTINCT c.document) FROM codes k JOIN chunks c ON c.id = k.chunk"
    (found,) = rows(transaction, Count, sql)
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


def refuse_recipe(transaction: Reading, model: str) -> None:
    found = recipe_of(transaction)
    if model_of(transaction) is not None and found != str(RECIPE):
        raise StoreUnavailableError(
            f"the store holds vectors sift made from passages another way (recipe {found or 'unknown'}, now {RECIPE})",
            EMBED_REMEDY.format(model=model),
            code=Problem.OTHER_MODEL,
        )


def refuse_model(transaction: Reading, model: str) -> None:
    found = model_of(transaction)
    if found not in {None, model}:
        raise StoreUnavailableError(
            f"the store holds vectors of {found}, not of {model}",
            EMBED_REMEDY.format(model=model),
            code=Problem.OTHER_MODEL,
        )
    refuse_recipe(transaction, model)


def _refuse_another(transaction: Writing, model: str, width: int) -> None:
    refuse_model(transaction, model)
    held = width_held(transaction)
    if held not in {None, width}:
        raise StoreUnavailableError(
            f"the store holds vectors of {held:,} dimensions, not of {width:,}",
            EMBED_REMEDY.format(model=model),
            code=Problem.OTHER_MODEL,
        )


def put_vectors(transaction: Writing, chunks: Ids, matrix: Halves, model: str) -> VectorsPut:
    _checked(chunks, matrix)
    _refuse_another(transaction, model, width_of(matrix))
    wanted = json.dumps(ints(chunks))
    sql = "SELECT id FROM chunks WHERE id IN (SELECT value FROM json_each(?))"
    there = among(
        chunks, ascending(np.array([found.id for found in rows(transaction, IdRow, sql, (wanted,))], np.int64))
    )
    intact = np.isfinite(matrix).all(axis=1)
    kept, rows_kept = ints(chunks[there & intact]), matrix[there & intact]
    if kept:
        run_each(
            transaction,
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            [(MODEL_KEY, model), (WIDTH_KEY, str(width_of(matrix))), (RECIPE_KEY, str(RECIPE))],
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
    return VectorsPut(len(kept), tuple(ints(chunks[there & ~intact])))


def _rows(matrix: NDArray[np.generic]) -> list[bytes]:
    data = matrix.tobytes()
    size = len(data) // len(matrix) if len(matrix) else 0
    return [data[at : at + size] for at in range(0, len(data), size)] if size else []


def forget_vectors(transaction: Writing) -> None:
    run(transaction, "DELETE FROM vectors")
    run(transaction, "DELETE FROM meta WHERE key IN (?, ?, ?)", (MODEL_KEY, WIDTH_KEY, RECIPE_KEY))


def vector_generation(transaction: Reading) -> int:
    (held,) = rows(transaction, Count, "SELECT vector_changes FROM counts")
    return held.n


class _CodesHeld(NamedTuple):
    rows: int
    size: int


def _held(transaction: Reading) -> _CodesHeld:
    (held,) = rows(transaction, Count, "SELECT vectors FROM counts")
    return _CodesHeld(held.n, code_bytes(width_held(transaction) or 0))


def _batches(transaction: Reading, held: _CodesHeld, read: int) -> Iterator[tuple[int, Ids, bytes]]:
    done = 0
    cursor = transaction.connection.execute(CODES)
    while batch := cast("list[tuple[int, bytes]]", cursor.fetchmany(read)):
        if done + len(batch) > held.rows or any(len(row[1]) != held.size for row in batch):
            raise StoreUnavailableError(DAMAGED_CODES, CODES_REMEDY, code=Problem.STORE_DAMAGED)
        yield done, np.fromiter((row[0] for row in batch), np.int64, len(batch)), b"".join(row[1] for row in batch)
        done += len(batch)
    if done != held.rows:
        raise StoreUnavailableError(DAMAGED_CODES, CODES_REMEDY, code=Problem.STORE_DAMAGED)


def codes_of(transaction: Reading, read: int = READ) -> tuple[Ids, Bits]:
    held = _held(transaction)
    ids, bits = np.empty(held.rows, np.int64), np.empty((held.rows, held.size), np.uint8)
    for done, batch, joined in _batches(transaction, held, read):
        ids[done : done + len(batch)] = batch
        bits[done : done + len(batch)] = np.frombuffer(joined, np.uint8).reshape(-1, held.size)
    return ids, bits


def resident_codes_of(transaction: Reading, read: int = READ) -> tuple[Ids, HeldCodes]:
    held = _held(transaction)
    ids, made = np.empty(held.rows, np.int64), filling(held.rows, held.size)
    for done, batch, joined in _batches(transaction, held, read):
        ids[done : done + len(batch)] = batch
        filled(made, joined)
    return ids, finished(made)


def vectors_of(transaction: Reading, chunks: Ids) -> tuple[Ids, Halves]:
    asked = ints(chunks)
    touched(transaction, TOUCH_VECTORS, asked)
    found = rows(transaction, VectorRow, VECTORS, (json.dumps(asked),))
    width = width_held(transaction) or 0
    if any(len(row.vector) != 2 * width for row in found):
        raise DamagedError(f"a vector of the store is not of its {width:,} dimensions")
    ids = np.fromiter((row.chunk for row in found), np.int64, len(found))
    stored = np.frombuffer(b"".join(row.vector for row in found), STORED_HALF)
    return ids, stored.astype(np.float16).reshape(len(found), width)
