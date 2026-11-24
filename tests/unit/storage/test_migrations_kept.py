import contextlib
from pathlib import Path
from typing import NamedTuple

import pytest

from sift.api.ids import CollectionId, DocumentId, first_row
from sift.storage.database import Writing, reading, rows, run, run_each, writing
from sift.storage.documents import folded, folded_names
from sift.storage.layout import LAYOUT
from sift.storage.store import create_store, open_store
from sift.storage.texts import put_text, text_of, texts_cut

COLLECTION = CollectionId(1)
PATH = "Notizen/Äpfel.md"
TEXT = "Äpfel und Birnen\nzweite Zeile\n"
FIELDS = '{"tags": ["obst"], "title": "Äpfel"}'
FIRST = first_row(COLLECTION) + 1
SECOND, THIRD = FIRST + 1, FIRST + 2
DOCUMENT_WITH_TEXT = """INSERT INTO documents(id, source, key, digest, title, fields, text)
                        VALUES (?, 1, '', 'd', 'Äpfel', ?, ?)"""
DOCUMENT = "INSERT INTO documents(id, source, key, digest, title, fields) VALUES (?, 1, '', 'd', 'Äpfel', ?)"
CHUNK = "INSERT INTO chunks(document, first_line, last_line, start, length) VALUES (?, ?, ?, ?, ?)"
LINK = "INSERT INTO links(document, ordinal, line, kind, target, key) VALUES (?, 0, 1, 'wikilink', 'B', 'b')"
SEGMENT_WITH_TOKENS = """INSERT INTO keyword_segments(id, first, last, chunks, dead, tokens, average)
                         VALUES (1, ?, ?, 2, 1, 7, 3.5)"""
SEGMENT = "INSERT INTO keyword_segments(id, first, last, chunks, dead, average) VALUES (1, ?, ?, 2, 1, 3.5)"
COUNTS = """SELECT documents, chunks, vectors, vectors_written, (SELECT count(*) FROM documents),
                   (SELECT count(*) FROM chunks), (SELECT count(*) FROM vectors) FROM counts"""


def _made_by(found: str) -> int:
    return 1 + next(
        at for at, migration in enumerate(LAYOUT) if any(isinstance(step, str) and found in step for step in migration)
    )


def _made_by_step(step: object) -> int:
    return 1 + next(at for at, migration in enumerate(LAYOUT) if step in migration)


KEYWORDS = _made_by("CREATE TABLE keyword_segments(")
VECTORS = _made_by("CREATE TABLE vectors(")
NO_TOKENS = _made_by("DROP COLUMN tokens")
TEXTS = _made_by_step(texts_cut)
FOLDED = _made_by_step(folded_names)


class Problem(NamedTuple):
    found: str


class Counts(NamedTuple):
    documents: int
    chunks: int
    vectors: int
    vectors_written: int
    held_documents: int
    held_chunks: int
    held_vectors: int


class Placed(NamedTuple):
    id: int
    place: int | None


class Chunk(NamedTuple):
    chunk: int


class Source(NamedTuple):
    path: str
    folded: str


def _document(transaction: Writing, at: int) -> None:
    run(transaction, "INSERT INTO sources(id, path, modified_ns, size) VALUES (1, ?, 1, 40)", (PATH,))
    if at >= FOLDED:
        run(transaction, "UPDATE sources SET folded = ? WHERE id = 1", (folded(PATH),))
    if at < TEXTS:
        run(transaction, DOCUMENT_WITH_TEXT, (FIRST, FIELDS, TEXT))
    else:
        run(transaction, DOCUMENT, (FIRST, FIELDS))
        put_text(transaction, DocumentId(FIRST), (TEXT,))


def _filled(transaction: Writing, at: int) -> None:
    _document(transaction, at)
    run_each(transaction, CHUNK, [(FIRST, 1, 1, 0, 16), (FIRST, 2, 2, 17, 12)])
    run(transaction, LINK, (FIRST,))
    run(transaction, "INSERT INTO names(key, document, kind) VALUES ('äpfel', ?, 'name')", (FIRST,))
    run(transaction, "INSERT INTO unreadable(path, problem, read_again) VALUES ('kaputt.pdf', 'damaged', 1)")
    if at >= KEYWORDS:
        run(transaction, SEGMENT_WITH_TOKENS if at < NO_TOKENS else SEGMENT, (FIRST, SECOND))
        run(transaction, "INSERT INTO keyword_terms(term, chunks) VALUES ('äpfel', 1)")
        run(transaction, "INSERT INTO keyword_dead VALUES (?)", (FIRST,))
    if at >= VECTORS:
        run(transaction, "INSERT INTO vectors(chunk, vector) VALUES (?, x'00')", (FIRST,))


def _kept(transaction: Writing, at: int) -> None:
    assert rows(transaction, Problem, "PRAGMA integrity_check") == [Problem("ok")]
    assert rows(transaction, Problem, "PRAGMA foreign_key_check") == []
    (counts,) = rows(transaction, Counts, COUNTS)
    held = int(at >= VECTORS)
    assert counts == Counts(1, 2, held, held, 1, 2, held)
    unembedded = [Chunk(SECOND)] if held else [Chunk(FIRST), Chunk(SECOND)]
    assert rows(transaction, Chunk, "SELECT chunk FROM unembedded ORDER BY chunk") == unembedded
    placed = rows(transaction, Placed, "SELECT id, place FROM chunks ORDER BY id")
    assert placed == [Placed(FIRST, FIRST), Placed(SECOND, SECOND)]
    assert rows(transaction, Source, "SELECT path, folded FROM sources") == [Source(PATH, folded(PATH))]
    assert text_of(transaction, DocumentId(FIRST)) == TEXT
    assert rows(transaction, Chunk, "SELECT place FROM keyword_dead") == [Chunk(FIRST)] * int(at >= KEYWORDS)


@pytest.mark.parametrize("at", range(1, len(LAYOUT)))
def test_every_migration_keeps_a_store_s_data_and_counts(tmp_path: Path, at: int) -> None:
    path, older = tmp_path / "s.sqlite", LAYOUT[:at]
    create_store(path, COLLECTION, older)
    with contextlib.closing(open_store(path, COLLECTION, older)) as connection, writing(connection) as transaction:
        _filled(transaction, at)
    with contextlib.closing(open_store(path, COLLECTION)) as connection:
        with writing(connection) as transaction:
            _kept(transaction, at)
            run(transaction, CHUNK, (FIRST, 2, 2, 0, 1))
        with reading(connection) as transaction:
            assert rows(transaction, Placed, "SELECT id, place FROM chunks WHERE id = ?", (THIRD,)) == [
                Placed(THIRD, THIRD)
            ]
            assert Chunk(THIRD) in rows(transaction, Chunk, "SELECT chunk FROM unembedded")
