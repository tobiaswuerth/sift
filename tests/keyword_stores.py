import contextlib
from collections.abc import Generator
from pathlib import Path
from typing import NamedTuple

from sift.api.ids import CollectionId, DocumentId
from sift.domain.codec.shape import Shape
from sift.domain.text.notes import note
from sift.storage.database import BATCHES, Batches, Connection, Reading, Writing, WritingDocuments, rows, run
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


TEXTS = """SELECT c.id AS chunk, c.document AS document, d.title AS title,
                  substr(d.text, c.start + 1, c.length) AS text, d.meta AS meta
           FROM chunks c JOIN keyword_documents d ON d.id = c.document ORDER BY c.id"""
ORACLE = """CREATE VIRTUAL TABLE temp.oracle USING fts5(title, text, meta, tokenize='unicode61 remove_diacritics 2')"""
FILLED = """INSERT INTO temp.oracle(rowid, title, text, meta)
            SELECT c.id, d.title, substr(d.text, c.start + 1, c.length), d.meta
            FROM chunks c JOIN keyword_documents d ON d.id = c.document"""


def texts(transaction: Reading) -> list[Text]:
    return rows(transaction, Text, TEXTS)


def instances(transaction: Reading) -> list[Instance]:
    as_writer = Writing(transaction.connection)
    run(as_writer, ORACLE)
    run(as_writer, FILLED)
    run(as_writer, "CREATE VIRTUAL TABLE temp.instances USING fts5vocab(temp, oracle, instance)")
    found = rows(transaction, Instance, "SELECT term, doc, col, offset FROM temp.instances ORDER BY doc, col, offset")
    run(as_writer, "DROP TABLE temp.instances")
    run(as_writer, "DROP TABLE temp.oracle")
    return found
