import json
from collections.abc import Sequence
from typing import NamedTuple

from sift.api.ids import DocumentId
from sift.storage.database import KeyRow, Reading, Writing, rows, run
from sift.storage.documents import KeptDocument, SourceId

TAKEN = "CREATE TEMP TABLE IF NOT EXISTS taken(key TEXT PRIMARY KEY) STRICT"
TAKE = """INSERT INTO taken(key) SELECT value FROM json_each(:keys) WHERE true
          ON CONFLICT(key) DO NOTHING RETURNING key"""
KEYED = """SELECT id, key, digest FROM documents
           WHERE source = :source AND key IN (SELECT value FROM json_each(:keys))"""
UNTAKEN = """SELECT id FROM documents WHERE source = ? AND key NOT IN (SELECT key FROM temp.taken) ORDER BY id"""


class DocumentRow(NamedTuple):
    id: DocumentId


def nothing_taken(transaction: Writing) -> None:
    run(transaction, TAKEN)
    run(transaction, "DELETE FROM temp.taken")


def take(transaction: Writing, keys: Sequence[str]) -> frozenset[str]:
    return frozenset(each.key for each in rows(transaction, KeyRow, TAKE, {"keys": json.dumps(list(keys))}))


def keyed(transaction: Reading, source: SourceId, keys: Sequence[str]) -> dict[str, KeptDocument]:
    found = rows(transaction, KeptDocument, KEYED, {"source": source, "keys": json.dumps(list(keys))})
    return {each.key: each for each in found}


def untaken(transaction: Reading, source: SourceId) -> list[DocumentId]:
    return [each.id for each in rows(transaction, DocumentRow, UNTAKEN, (source,))]
