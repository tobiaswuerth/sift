import json
from collections.abc import Mapping, Sequence
from typing import NamedTuple, NewType, cast

from sift.api.fields import Fields
from sift.api.ids import ChunkId, DocumentId
from sift.domain.text.links import names
from sift.domain.text.notes import Document
from sift.storage.database import Reading, Writing, first, inserted, rows, run, run_each

SourceId = NewType("SourceId", int)


class Source(NamedTuple):
    id: SourceId
    path: str
    modified_ns: int
    size: int


class Kept(NamedTuple):
    id: DocumentId
    key: str
    digest: str


class Stored(NamedTuple):
    id: DocumentId
    path: str
    key: str
    digest: str
    title: str
    fields: str
    text: str


class StoredChunk(NamedTuple):
    id: ChunkId
    first_line: int
    last_line: int
    start: int
    length: int


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(cast("Mapping[str, object]", value))
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return list(value)
    raise TypeError(f"{type(value).__name__} is no field value")


def fields_json(fields: Fields) -> str:
    return json.dumps(fields, ensure_ascii=False, allow_nan=False, default=_plain)


def fields_of(written: str) -> Fields:
    return cast("Fields", json.loads(written))


def sources(transaction: Reading) -> dict[str, Source]:
    found = rows(transaction, Source, "SELECT id, path, modified_ns, size FROM sources")
    return {source.path: source for source in found}


class Identity(NamedTuple):
    id: SourceId


def put_source(transaction: Writing, path: str, modified_ns: int, size: int) -> SourceId:
    sql = """INSERT INTO sources(path, modified_ns, size) VALUES (?, ?, ?)
             ON CONFLICT(path) DO UPDATE SET modified_ns = excluded.modified_ns, size = excluded.size RETURNING id"""
    (written,) = rows(transaction, Identity, sql, (path, modified_ns, size))
    return written.id


def delete_source(transaction: Writing, path: str) -> None:
    run(transaction, "DELETE FROM sources WHERE path = ?", (path,))


def kept(transaction: Reading, source: SourceId) -> list[Kept]:
    return rows(transaction, Kept, "SELECT id, key, digest FROM documents WHERE source = ? ORDER BY id", (source,))


def _add(transaction: Writing, source: SourceId, path: str, document: Document) -> DocumentId:
    identity = DocumentId(
        inserted(
            transaction,
            "INSERT INTO documents(source, key, digest, title, fields, text) VALUES (?, ?, ?, ?, ?, ?)",
            (source, document.key, document.digest, document.title, fields_json(document.fields), document.text),
        )
    )
    run_each(
        transaction,
        "INSERT INTO chunks(document, first_line, last_line, start, length) VALUES (?, ?, ?, ?, ?)",
        [(identity, c.first_line, c.last_line, c.start, c.length) for c in document.chunks],
    )
    run_each(
        transaction,
        """INSERT INTO links(document, ordinal, field, line, kind, target, key, fragment, shown)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (identity, ordinal, link.field, link.line, link.kind, link.target, link.key, link.fragment, link.shown)
            for ordinal, link in enumerate(document.links)
        ],
    )
    if not document.key:
        run_each(
            transaction,
            "INSERT INTO names(key, document, kind) VALUES (?, ?, ?)",
            [(name.key, identity, name.kind) for name in names(path, document.fields)],
        )
    return identity


def add_documents(
    transaction: Writing, source: SourceId, path: str, documents: Sequence[Document]
) -> tuple[DocumentId, ...]:
    return tuple(_add(transaction, source, path, document) for document in documents)


def delete_documents(transaction: Writing, documents: Sequence[DocumentId]) -> None:
    run(
        transaction,
        "DELETE FROM documents WHERE id IN (SELECT value FROM json_each(?))",
        (json.dumps(list(documents)),),
    )


def stored(transaction: Reading, path: str, key: str = "") -> Stored | None:
    return first(
        transaction,
        Stored,
        """SELECT d.id, s.path, d.key, d.digest, d.title, d.fields, d.text
           FROM documents d JOIN sources s ON s.id = d.source WHERE s.path = ? AND d.key = ?""",
        (path, key),
    )


def chunks_of(transaction: Reading, document: DocumentId) -> list[StoredChunk]:
    sql = "SELECT id, first_line, last_line, start, length FROM chunks WHERE document = ? ORDER BY id"
    return rows(transaction, StoredChunk, sql, (document,))


class Counts(NamedTuple):
    documents: int
    chunks: int


def counts(transaction: Reading) -> Counts:
    (kept_counts,) = rows(transaction, Counts, "SELECT documents, chunks FROM counts")
    return kept_counts
