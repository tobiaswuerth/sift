import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

from sift.api.ids import DocumentId
from sift.domain.text.notes import note
from sift.storage.database import Connection, Count, IdRow, Reading, reading, rows, writing
from sift.storage.documents import add_documents, kept, put_source
from sift.storage.forgetting import delete_documents
from sift.storage.kept_vectors import AT_ONCE, vectors_held, vectors_kept
from sift.storage.keywords import writing_documents
from sift.storage.vectors import left, pending, put_vectors
from tests.keyword_stores import SMALL, chunk_texts, opened_store

LONG = "# Alpha\n\n" + "".join(f"Paragraph {number:04} about brushes and strokes.\n\n" for number in range(3000))


class Held(NamedTuple):
    chunk: int
    vector: bytes
    bits: bytes


class Kept(NamedTuple):
    text: str
    vector: bytes
    bits: bytes


HELD = "SELECT v.chunk, v.vector, k.bits FROM vectors v JOIN codes k ON k.chunk = v.chunk ORDER BY v.chunk"


def held(transaction: Reading) -> list[Kept]:
    cut = chunk_texts(transaction)
    return [Kept(cut[each.chunk], each.vector, each.bits) for each in rows(transaction, Held, HELD)]


def written_again(connection: Connection, text: str) -> None:
    with writing_documents(connection) as transaction:
        source = put_source(transaction, "a.md", 1, 1)
        old = [DocumentId(each.id) for each in kept(transaction, source)]
        vectors_held(transaction, old)
        delete_documents(transaction, old)
        vectors_kept(transaction, add_documents(transaction, source, "a.md", [note(text.encode(), "a.md")]))


def test_a_document_written_again_keeps_the_vectors_of_its_chunks_whose_embedded_text_is_unchanged() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        written_again(connection, LONG)
        with reading(connection) as transaction:
            ids = np.array([each.chunk for each in pending(transaction, 0, 10_000)], np.int64)
        assert len(ids) > AT_ONCE // 100
        with writing(connection) as transaction:
            put_vectors(transaction, ids, np.random.default_rng(0).normal(size=(len(ids), 16)).astype(np.float16), "m")
        with reading(connection) as transaction:
            before = {each.text: each for each in held(transaction)}
        written_again(connection, LONG.replace("Paragraph 1500 ", "Paragraph XVCD "))
        with reading(connection) as transaction:
            after = held(transaction)
            texts = list(chunk_texts(transaction).values())
            assert rows(transaction, Count, "SELECT count(*) FROM temp.kept_vectors") == [Count(0)]
            assert left(transaction) == len([text for text in texts if text not in before]) > 0
        assert all(before[each.text] == each for each in after)
        assert len(after) == len(texts) - left_of(connection) > 0


def left_of(connection: Connection) -> int:
    with reading(connection) as transaction:
        return left(transaction)


def test_nothing_held_keeps_nothing() -> None:
    with tempfile.TemporaryDirectory() as folder, opened_store(Path(folder), 3, SMALL) as connection:
        written_again(connection, "# Alpha\n\none\n")
        with writing_documents(connection) as transaction:
            vectors_held(transaction, [])
            documents = [DocumentId(each.id) for each in rows(transaction, IdRow, "SELECT id FROM documents")]
            vectors_kept(transaction, documents)
        with reading(connection) as transaction:
            assert left(transaction) == 1
