import json
from collections.abc import Mapping, Sequence
from typing import NamedTuple, NewType, cast

from sift.api.fields import Fields
from sift.api.ids import ChunkId, DocumentId
from sift.domain.errors import DamagedError
from sift.domain.text.links import names
from sift.domain.text.notes import Document
from sift.storage.database import Reading, Writing, WritingDocuments, first, inserted, rows, run, run_each
from sift.storage.keywords import forget

SourceId = NewType("SourceId", int)


class Source(NamedTuple):
    id: SourceId
    path: str
    modified_ns: int
    size: int


class KeptDocument(NamedTuple):
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
    read = cast("object", json.loads(written))
    if not isinstance(read, dict):
        raise DamagedError("a document's front matter is no mapping")
    return cast("Fields", read)


def sources(transaction: Reading) -> dict[str, Source]:
    found = rows(transaction, Source, "SELECT id, path, modified_ns, size FROM sources")
    return {source.path: source for source in found}


class SourceRow(NamedTuple):
    id: SourceId


def put_source(transaction: Writing, path: str, modified_ns: int, size: int) -> SourceId:
    sql = """INSERT INTO sources(path, modified_ns, size) VALUES (?, ?, ?)
             ON CONFLICT(path) DO UPDATE SET modified_ns = excluded.modified_ns, size = excluded.size RETURNING id"""
    (written,) = rows(transaction, SourceRow, sql, (path, modified_ns, size))
    return written.id


class Owned(NamedTuple):
    id: DocumentId


def delete_sources(transaction: WritingDocuments, paths: Sequence[str]) -> int:
    chosen = (json.dumps(list(paths)),)
    held = rows(
        transaction,
        Owned,
        """SELECT id FROM documents
           WHERE source IN (SELECT id FROM sources WHERE path IN (SELECT value FROM json_each(?)))""",
        chosen,
    )
    forget(transaction, [document.id for document in held])
    run(transaction, "DELETE FROM sources WHERE path IN (SELECT value FROM json_each(?))", chosen)
    return len(held)


def kept(transaction: Reading, source: SourceId) -> list[KeptDocument]:
    return rows(
        transaction, KeptDocument, "SELECT id, key, digest FROM documents WHERE source = ? ORDER BY id", (source,)
    )


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
    transaction: WritingDocuments, source: SourceId, path: str, documents: Sequence[Document]
) -> tuple[DocumentId, ...]:
    return tuple(_add(transaction, source, path, document) for document in documents)


def delete_documents(transaction: WritingDocuments, documents: Sequence[DocumentId]) -> None:
    forget(transaction, documents)
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


class PathRow(NamedTuple):
    path: str


PATHS_NAMED = """SELECT s.path FROM sources s JOIN documents d ON d.source = s.id AND d.key = ''
                 WHERE s.name = ? COLLATE NOCASE ORDER BY s.path"""


def paths_named(transaction: Reading, name: str) -> list[str]:
    return [found.path for found in rows(transaction, PathRow, PATHS_NAMED, (name,))]


class Passage(NamedTuple):
    chunk: ChunkId
    document: DocumentId
    path: str
    key: str
    title: str
    first_line: int
    last_line: int
    start: int
    length: int


class TextRow(NamedTuple):
    id: DocumentId
    text: str


PASSAGES = """SELECT c.id, d.id, s.path, d.key, d.title, c.first_line, c.last_line, c.start, c.length
              FROM chunks c JOIN documents d ON d.id = c.document JOIN sources s ON s.id = d.source
              WHERE c.id IN (SELECT value FROM json_each(?))"""


def passages(transaction: Reading, chunks: Sequence[ChunkId]) -> dict[ChunkId, Passage]:
    return {found.chunk: found for found in rows(transaction, Passage, PASSAGES, (json.dumps(list(chunks)),))}


class FieldsRow(NamedTuple):
    id: int
    fields: str


def fields_by_document(transaction: Reading, documents: Sequence[int]) -> dict[int, Fields]:
    sql = "SELECT id, fields FROM documents WHERE id IN (SELECT value FROM json_each(?))"
    return {
        found.id: fields_of(found.fields) for found in rows(transaction, FieldsRow, sql, (json.dumps(list(documents)),))
    }


class ChunkOwner(NamedTuple):
    chunk: int
    document: int


def documents_of(transaction: Reading, chunks: Sequence[int]) -> dict[int, int]:
    sql = "SELECT id, document FROM chunks WHERE id IN (SELECT value FROM json_each(?))"
    return {found.chunk: found.document for found in rows(transaction, ChunkOwner, sql, (json.dumps(list(chunks)),))}


def texts(transaction: Reading, documents: Sequence[DocumentId]) -> dict[DocumentId, str]:
    sql = "SELECT id, text FROM documents WHERE id IN (SELECT value FROM json_each(?))"
    return {found.id: found.text for found in rows(transaction, TextRow, sql, (json.dumps(list(documents)),))}


class Unreadable(NamedTuple):
    path: str
    problem: str
    read_again: int


def mark_unreadable(transaction: Writing, path: str, problem: str, *, read_again: bool) -> None:
    sql = "INSERT OR REPLACE INTO unreadable(path, problem, read_again) VALUES (?, ?, ?)"
    run(transaction, sql, (path, problem, int(read_again)))


def clear_unreadable(transaction: Writing, paths: Sequence[str]) -> None:
    chosen = (json.dumps(list(paths)),)
    run(transaction, "DELETE FROM unreadable WHERE path IN (SELECT value FROM json_each(?))", chosen)


def unreadable(transaction: Reading) -> list[Unreadable]:
    return rows(transaction, Unreadable, "SELECT path, problem, read_again FROM unreadable ORDER BY path")


class Counts(NamedTuple):
    documents: int
    chunks: int


def counts(transaction: Reading) -> Counts:
    (kept_counts,) = rows(transaction, Counts, "SELECT documents, chunks FROM counts")
    return kept_counts
