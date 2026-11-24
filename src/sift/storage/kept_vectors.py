import hashlib
import json
from collections.abc import Iterator, Sequence
from typing import NamedTuple

from sift.api.ids import ChunkId, DocumentId
from sift.domain.text.embedding import embedded
from sift.storage.database import Writing, batches, first, run, run_each
from sift.storage.texts import TextSpan, spans_text

AT_ONCE = 1000
KEY_BYTES = 16
KEPT = """CREATE TEMP TABLE IF NOT EXISTS kept_vectors(
              key BLOB PRIMARY KEY, vector BLOB NOT NULL, bits BLOB NOT NULL) WITHOUT ROWID"""
HELD = """SELECT c.id, c.document, c.start, c.length, d.title,
                 c.id = (SELECT min(f.id) FROM chunks f WHERE f.document = c.document), v.vector, k.bits
          FROM chunks c JOIN documents d ON d.id = c.document
               JOIN vectors v ON v.chunk = c.id JOIN codes k ON k.chunk = c.id
          WHERE c.document IN (SELECT value FROM json_each(?)) ORDER BY c.id"""
PLACED = """SELECT c.id, c.document, c.start, c.length, d.title,
                   c.id = (SELECT min(f.id) FROM chunks f WHERE f.document = c.document), NULL, NULL
            FROM chunks c JOIN documents d ON d.id = c.document
            WHERE c.document IN (SELECT value FROM json_each(?)) ORDER BY c.id"""
VECTORS_KEPT = "INSERT INTO vectors(chunk, vector) SELECT ?, vector FROM temp.kept_vectors WHERE key = ?"
CODES_KEPT = "INSERT INTO codes(chunk, bits) SELECT ?, bits FROM temp.kept_vectors WHERE key = ?"


class _Chunk(NamedTuple):
    chunk: ChunkId
    document: DocumentId
    start: int
    length: int
    title: str
    first: int
    vector: bytes | None
    bits: bytes | None


class _Any(NamedTuple):
    found: int


def _key(title: str, text: str, *, first: bool) -> bytes:
    held = embedded(title, text, first=first).encode(errors="surrogatepass")
    return hashlib.blake2b(held, digest_size=KEY_BYTES).digest()


def _keyed(transaction: Writing, sql: str, documents: Sequence[DocumentId]) -> Iterator[list[tuple[_Chunk, bytes]]]:
    for batch in batches(transaction, _Chunk, sql, (json.dumps(list(documents)),), AT_ONCE):
        cut = spans_text(transaction, [TextSpan(each.document, each.start, each.length) for each in batch])
        yield [(each, _key(each.title, text, first=bool(each.first))) for each, text in zip(batch, cut, strict=True)]


def vectors_held(transaction: Writing, documents: Sequence[DocumentId]) -> None:
    run(transaction, KEPT)
    for batch in _keyed(transaction, HELD, documents):
        run_each(
            transaction,
            "INSERT OR IGNORE INTO temp.kept_vectors(key, vector, bits) VALUES (?, ?, ?)",
            [(key, each.vector, each.bits) for each, key in batch],
        )


def vectors_kept(transaction: Writing, documents: Sequence[DocumentId]) -> None:
    run(transaction, KEPT)
    if first(transaction, _Any, "SELECT 1 FROM temp.kept_vectors LIMIT 1") is None:
        return
    for batch in _keyed(transaction, PLACED, documents):
        placed = [(each.chunk, key) for each, key in batch]
        run_each(transaction, VECTORS_KEPT, placed)
        run_each(transaction, CODES_KEPT, placed)


def vectors_let_go(transaction: Writing) -> None:
    run(transaction, KEPT)
    run(transaction, "DELETE FROM temp.kept_vectors")
