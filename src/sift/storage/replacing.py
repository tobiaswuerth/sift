from collections.abc import Mapping, Sequence

from sift.domain.plan.changes import Changes, changes
from sift.domain.text.notes import Document
from sift.storage.database import WritingDocuments
from sift.storage.documents import KeptDocument, SourceId, add_documents
from sift.storage.forgetting import delete_documents
from sift.storage.kept_vectors import vectors_held, vectors_kept


def documents_replaced(
    transaction: WritingDocuments,
    source: SourceId,
    path: str,
    held: Mapping[str, KeptDocument],
    read: Sequence[Document],
) -> Changes:
    found = changes({key: each.digest for key, each in held.items()}, [(each.key, each.digest) for each in read])
    by_key = {each.key: each for each in read}
    replaced = [held[key].id for key in found.changed]
    vectors_held(transaction, replaced)
    delete_documents(transaction, [*(held[key].id for key in found.gone), *replaced])
    added = add_documents(transaction, source, path, [by_key[key] for key in (*found.changed, *found.new)])
    vectors_kept(transaction, added)
    return found
