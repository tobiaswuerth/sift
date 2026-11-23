import json
from collections.abc import Sequence
from typing import NamedTuple

from sift.api.ids import DocumentId
from sift.storage.database import WritingDocuments, rows, run
from sift.storage.keywords import forget


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


def delete_documents(transaction: WritingDocuments, documents: Sequence[DocumentId]) -> None:
    forget(transaction, documents)
    run(
        transaction,
        "DELETE FROM documents WHERE id IN (SELECT value FROM json_each(?))",
        (json.dumps(list(documents)),),
    )
