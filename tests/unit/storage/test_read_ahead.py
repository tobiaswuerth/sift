import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

from sift.api.fields import Fields
from sift.api.ids import ChunkId
from sift.domain.arrays import ints
from sift.storage.catalog import known_collections
from sift.storage.columns import documents_of
from sift.storage.database import IdRow, Reading, reading, rows
from sift.storage.documents import Passage, fields_by_document, passages
from sift.storage.store import read_connected
from sift.storage.texts import TextSpan, spans_text
from sift.storage.vectors import vectors_of
from tests.homes import home_of
from tests.models import Hashed

NOTES = {
    f"notes/n{n}.md": f"---\ntopic: t{n}\n---\n# Note {n}\n\n" + "a passage of words. " * (40 * n) for n in range(1, 9)
}


@dataclass(frozen=True, slots=True)
class Recorded:
    asked: list[tuple[str, list[object]]] = field(default_factory=list[tuple[str, list[object]]])

    def touched(self, sql: str, keys: Sequence[object]) -> None:
        self.asked.append((sql, list(keys)))


@dataclass(frozen=True, slots=True)
class Read:
    chunks: list[int]
    vectors: list[list[float]]
    owners: list[int]
    passages: dict[ChunkId, Passage]
    texts: list[str]
    fields: dict[int, Fields]


def read_all(transaction: Reading) -> Read:
    chunks = np.array([each.id for each in rows(transaction, IdRow, "SELECT id FROM chunks ORDER BY id")], np.int64)
    ids, vectors = vectors_of(transaction, chunks)
    owners = documents_of(transaction, chunks)
    held = passages(transaction, [ChunkId(chunk) for chunk in ints(chunks)])
    texts = spans_text(transaction, [TextSpan(each.document, each.start, each.length) for each in held.values()])
    fields = fields_by_document(transaction, sorted({each.document for each in held.values()}))
    return Read(
        ints(ids),
        cast("list[list[float]]", vectors.astype(np.float64).tolist()),
        ints(owners),
        dict(held),
        texts,
        dict(fields),
    )


def test_a_search_s_reads_of_many_rows_read_them_ahead_and_find_what_they_find_alone(tmp_path: Path) -> None:
    (one,) = known_collections(home := home_of(tmp_path, NOTES, embedded=Hashed()))
    recorded = Recorded()
    with read_connected(one.store, one.id) as connection:
        with reading(connection) as alone:
            plain = read_all(alone)
        with reading(connection, recorded) as beside:
            ahead = read_all(beside)
        found = [connection.execute(sql, (json.dumps(keys),)).fetchone()[0] for sql, keys in recorded.asked]
    assert (ahead, home.is_dir()) == (plain, True)
    chunks, documents = len(plain.chunks), len(plain.fields)
    assert chunks > documents > 1
    assert [len(keys) for _, keys in recorded.asked[:3]] == [chunks, chunks, chunks]
    assert {cast("list[int]", pair)[0] for pair in recorded.asked[3][1]} == {
        each.document for each in plain.passages.values()
    }
    assert len(recorded.asked[4][1]) == documents
    assert found[:3] == [2 * Hashed().width * chunks, chunks, chunks]
    assert found[3] >= sum(len(text) for text in plain.texts) > 0
    assert found[4] > 0
