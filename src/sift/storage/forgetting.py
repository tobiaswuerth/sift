import json
from collections.abc import Sequence
from typing import NamedTuple

from sift.api.ids import DocumentId
from sift.storage.database import Reading, WritingDocuments, rows, run
from sift.storage.keywords import forget


class Owned(NamedTuple):
    id: DocumentId


def documents_at(transaction: Reading, paths: Sequence[str]) -> list[DocumentId]:
    held = rows(
        transaction,
        Owned,
        """SELECT id FROM documents
           WHERE source IN (SELECT id FROM sources WHERE path IN (SELECT value FROM json_each(?)))""",
        (json.dumps(list(paths)),),
    )
    return [document.id for document in held]


def delete_sources(transaction: WritingDocuments, paths: Sequence[str]) -> int:
    held = documents_at(transaction, paths)
    forget(transaction, held)
    run(transaction, "DELETE FROM sources WHERE path IN (SELECT value FROM json_each(?))", (json.dumps(list(paths)),))
    return len(held)


def delete_documents(transaction: WritingDocuments, documents: Sequence[DocumentId]) -> None:
    forget(transaction, documents)
    run(
        transaction,
        "DELETE FROM documents WHERE id IN (SELECT value FROM json_each(?))",
        (json.dumps(list(documents)),),
    )
