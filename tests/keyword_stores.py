import contextlib
from collections.abc import Generator
from pathlib import Path
from typing import NamedTuple

from sift.api.ids import CollectionId, DocumentId
from sift.domain.codec.shape import Shape
from sift.domain.text.notes import note
from sift.storage.database import BATCHES, Batches, Connection, Reading, Writing, WritingDocuments, rows, run, run_each
from sift.storage.documents import add_documents, kept, put_source
from sift.storage.forgetting import delete_documents, delete_sources
from sift.storage.keywords import writing_documents
from sift.storage.store import create_store, open_store

SMALL = Shape(range_shift=2, block=2, long=3, lengths_shift=3)
COLUMNS = ("title", "text", "meta")

type Content = tuple[str, str, str]
type Round = dict[str, Content | None]


class Instance(NamedTuple):
    term: str
    chunk: int
    column: str
    offset: int


class Text(NamedTuple):
    chunk: int
    document: int
    title: str
    text: str
    meta: str | None


def written(fields: str, heading: str, body: str) -> bytes:
    head = f"---\n{fields}---\n" if fields else ""
    paragraphs = "\n\n".join(body[at : at + 120] for at in range(0, len(body), 120))
    return f"{head}# {heading}\n\n{paragraphs}\n".encode()


def changed(transaction: WritingDocuments, path: str, content: Content | None) -> None:
    if content is None:
        delete_sources(transaction, [path])
        return
    source = put_source(transaction, path, 0, 0)
    delete_documents(transaction, [DocumentId(held.id) for held in kept(transaction, source)])
    add_documents(transaction, source, path, [note(written(*content), path)])


def applied(connection: Connection, rounds: list[Round], batches: Batches = BATCHES) -> None:
    for each in rounds:
        with writing_documents(connection, batches) as transaction:
            for path, content in each.items():
                changed(transaction, path, content)


@contextlib.contextmanager
def opened_store(folder: Path, collection: int, shape: Shape) -> Generator[Connection]:
    path = folder / f"{collection}.sqlite"
    create_store(path, CollectionId(collection), shape=shape)
    with contextlib.closing(open_store(path, CollectionId(collection))) as connection:
        yield connection


PLACED = """SELECT c.id AS chunk, c.document AS document, d.title AS title, d.meta AS meta
            FROM chunks c JOIN keyword_documents d ON d.id = c.document ORDER BY c.id"""
ORACLE = """CREATE VIRTUAL TABLE temp.oracle USING fts5(title, text, meta, tokenize='unicode61 remove_diacritics 2')"""
FILLED = "INSERT INTO temp.oracle(rowid, title, text, meta) VALUES (?, ?, ?, ?)"


class Block(NamedTuple):
    document: int
    text: str


class Span(NamedTuple):
    id: int
    document: int
    start: int
    length: int


class Placed(NamedTuple):
    chunk: int
    document: int
    title: str
    meta: str | None


def chunk_texts(transaction: Reading) -> dict[int, str]:
    whole: dict[int, list[str]] = {}
    for block in rows(transaction, Block, "SELECT document, text FROM texts ORDER BY document, block"):
        whole.setdefault(block.document, []).append(block.text)
    joined = {document: "".join(parts) for document, parts in whole.items()}
    spans = rows(transaction, Span, "SELECT id, document, start, length FROM chunks ORDER BY id")
    return {span.id: joined.get(span.document, "")[span.start : span.start + span.length] for span in spans}


def texts(transaction: Reading) -> list[Text]:
    cut = chunk_texts(transaction)
    return [
        Text(each.chunk, each.document, each.title, cut[each.chunk], each.meta)
        for each in rows(transaction, Placed, PLACED)
    ]


def instances(transaction: Reading) -> list[Instance]:
    as_writer = Writing(transaction.connection)
    run(as_writer, ORACLE)
    run_each(as_writer, FILLED, [(each.chunk, each.title, each.text, each.meta) for each in texts(transaction)])
    run(as_writer, "CREATE VIRTUAL TABLE temp.instances USING fts5vocab(temp, oracle, instance)")
    found = rows(transaction, Instance, "SELECT term, doc, col, offset FROM temp.instances ORDER BY doc, col, offset")
    run(as_writer, "DROP TABLE temp.instances")
    run(as_writer, "DROP TABLE temp.oracle")
    return found


class ChunkVector(NamedTuple):
    chunk: int
    vector: bytes


def vectors_by_text(transaction: Reading) -> dict[str, bytes]:
    cut = chunk_texts(transaction)
    return {
        cut[each.chunk]: each.vector for each in rows(transaction, ChunkVector, "SELECT chunk, vector FROM vectors")
    }
