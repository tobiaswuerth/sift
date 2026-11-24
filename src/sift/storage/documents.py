import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, NewType, cast

from sift.api.fields import Fields
from sift.api.ids import ChunkId, DocumentId
from sift.domain.errors import DamagedError
from sift.domain.text.chunks import Chunk
from sift.domain.text.links import Link, names
from sift.domain.text.notes import Document, NotePart, NoteRead
from sift.storage.database import Reading, Writing, WritingDocuments, first, inserted, rows, run, run_each
from sift.storage.texts import put_text

SourceId = NewType("SourceId", int)
MOST_DESCRIPTION_BYTES = 512 << 20


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


def description_bytes(document: Document) -> int:
    return sum(len(text.encode()) for text in (document.key, document.title, fields_json(document.fields)))


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


def kept(transaction: Reading, source: SourceId) -> list[KeptDocument]:
    return rows(
        transaction, KeptDocument, "SELECT id, key, digest FROM documents WHERE source = ? ORDER BY id", (source,)
    )


CHUNK_ROW = "INSERT INTO chunks(document, first_line, last_line, start, length) VALUES (?, ?, ?, ?, ?)"
LINK_ROW = """INSERT INTO links(document, ordinal, field, line, kind, target, key, fragment, shown)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
NAME_ROW = "INSERT INTO names(key, document, kind) VALUES (?, ?, ?)"
DOCUMENT_ROW = "INSERT INTO documents(source, key, digest, title, fields) VALUES (?, ?, ?, ?, ?)"
WRITTEN_AT_ONCE = 1000


def _chunks_put(transaction: Writing, identity: DocumentId, chunks: Sequence[Chunk]) -> None:
    run_each(transaction, CHUNK_ROW, [(identity, c.first_line, c.last_line, c.start, c.length) for c in chunks])


def _links_put(transaction: Writing, identity: DocumentId, links: Sequence[Link], first: int) -> None:
    run_each(
        transaction,
        LINK_ROW,
        [
            (identity, ordinal, link.field, link.line, link.kind, link.target, link.key, link.fragment, link.shown)
            for ordinal, link in enumerate(links, first)
        ],
    )


def _names_put(transaction: Writing, identity: DocumentId, path: str, fields: Fields) -> None:
    run_each(transaction, NAME_ROW, [(name.key, identity, name.kind) for name in names(path, fields)])


def _add(transaction: Writing, source: SourceId, path: str, document: Document) -> DocumentId:
    row = (source, document.key, document.digest, document.title, fields_json(document.fields))
    identity = DocumentId(inserted(transaction, DOCUMENT_ROW, row))
    put_text(transaction, identity, (document.text,))
    _chunks_put(transaction, identity, document.chunks)
    _links_put(transaction, identity, document.links, 0)
    if not document.key:
        _names_put(transaction, identity, path, document.fields)
    return identity


@dataclass(slots=True)
class _Streamed:
    transaction: Writing
    identity: DocumentId
    chunks: list[Chunk] = field(default_factory=list[Chunk])
    links: list[Link] = field(default_factory=list[Link])
    linked: int = 0
    title: str = ""

    def flushed(self) -> None:
        _chunks_put(self.transaction, self.identity, self.chunks)
        _links_put(self.transaction, self.identity, self.links, self.linked)
        self.linked += len(self.links)
        self.chunks, self.links = [], []

    def texts(self, parts: Iterable[NotePart]) -> Iterator[str]:
        for part in parts:
            match part:
                case str():
                    yield part
                case Chunk():
                    self.chunks.append(part)
                case Link():
                    self.links.append(part)
                case _:
                    self.title = part.title
            if len(self.chunks) + len(self.links) >= WRITTEN_AT_ONCE:
                self.flushed()


def add_streamed(transaction: WritingDocuments, source: SourceId, path: str, read: NoteRead) -> DocumentId:
    identity = DocumentId(inserted(transaction, DOCUMENT_ROW, (source, "", "", "", fields_json(read.fields))))
    _names_put(transaction, identity, path, read.fields)
    streamed = _Streamed(transaction, identity)
    put_text(transaction, identity, streamed.texts(read.parts))
    streamed.flushed()
    run(transaction, "UPDATE documents SET title = ? WHERE id = ?", (streamed.title, identity))
    return identity


def set_digest(transaction: WritingDocuments, document: DocumentId, digested: str) -> None:
    run(transaction, "UPDATE documents SET digest = ? WHERE id = ?", (digested, document))


def add_documents(
    transaction: WritingDocuments, source: SourceId, path: str, documents: Sequence[Document]
) -> tuple[DocumentId, ...]:
    return tuple(_add(transaction, source, path, document) for document in documents)


def stored(transaction: Reading, path: str, key: str = "") -> Stored | None:
    return first(
        transaction,
        Stored,
        """SELECT d.id, s.path, d.key, d.digest, d.title, d.fields
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
    chunk: ChunkId
    document: DocumentId


def documents_of(transaction: Reading, chunks: Sequence[int]) -> dict[int, int]:
    sql = "SELECT id, document FROM chunks WHERE id IN (SELECT value FROM json_each(?))"
    return {found.chunk: found.document for found in rows(transaction, ChunkOwner, sql, (json.dumps(list(chunks)),))}


class Unreadable(NamedTuple):
    path: str
    problem: str
    read_again: int
    skipped_under: str | None
    stopped_under: str | None


MARKED = """INSERT OR REPLACE INTO unreadable(path, problem, read_again, skipped_under, stopped_under)
            VALUES (?, ?, ?, ?, ?)"""


@dataclass(frozen=True, slots=True)
class Mark:
    problem: str
    read_again: bool
    skipped_under: str | None = None
    stopped_under: str | None = None


def mark_unreadable(transaction: Writing, path: str, mark: Mark) -> None:
    run(transaction, MARKED, (path, mark.problem, int(mark.read_again), mark.skipped_under, mark.stopped_under))


def clear_unreadable(transaction: Writing, paths: Sequence[str]) -> None:
    chosen = (json.dumps(list(paths)),)
    run(transaction, "DELETE FROM unreadable WHERE path IN (SELECT value FROM json_each(?))", chosen)


def unreadable(transaction: Reading) -> list[Unreadable]:
    sql = "SELECT path, problem, read_again, skipped_under, stopped_under FROM unreadable ORDER BY path"
    return rows(transaction, Unreadable, sql)


class Counts(NamedTuple):
    documents: int
    chunks: int


def counts(transaction: Reading) -> Counts:
    (kept_counts,) = rows(transaction, Counts, "SELECT documents, chunks FROM counts")
    return kept_counts


FIELD_SAMPLE = 1_000
ENDS = "SELECT (SELECT min(id) FROM documents) AS first, (SELECT max(id) FROM documents) AS last"
SAMPLED_FIELDS = """SELECT DISTINCT f.key AS name FROM json_each(?) p,
    json_each((SELECT fields FROM documents WHERE id >= p.value ORDER BY id LIMIT 1)) f ORDER BY f.key"""


class Ends(NamedTuple):
    first: int | None
    last: int | None


class FieldName(NamedTuple):
    name: str


def fields_sampled(transaction: Reading, sample: int = FIELD_SAMPLE) -> tuple[str, ...]:
    (ends,) = rows(transaction, Ends, ENDS)
    if ends.first is None or ends.last is None:
        return ()
    steps = max(sample - 1, 1)
    ids = sorted({ends.first + (ends.last - ends.first) * at // steps for at in range(steps + 1)})
    return tuple(found.name for found in rows(transaction, FieldName, SAMPLED_FIELDS, (json.dumps(ids),)))


class KeyDigest(NamedTuple):
    key: str
    digest: str


def digests_of(transaction: Reading, path: str) -> dict[str, str]:
    sql = "SELECT d.key, d.digest FROM documents d JOIN sources s ON s.id = d.source WHERE s.path = ?"
    return {each.key: each.digest for each in rows(transaction, KeyDigest, sql, (path,))}
