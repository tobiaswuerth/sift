import hashlib
import posixpath
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from sift.api.ids import DocumentId
from sift.app.filing import Job, Tally
from sift.domain.plan.indexing import Stat
from sift.domain.text.decoding import text_pieces
from sift.domain.text.documents import Again, Refused
from sift.domain.text.notes import DIGEST_BYTES, read_note
from sift.ingest.files import Identity
from sift.storage.database import WritingDocuments, back_to_save_point, past_save_point, save_point
from sift.storage.documents import SourceId, add_streamed, kept, put_source, set_digest
from sift.storage.forgetting import delete_documents
from sift.storage.kept_vectors import vectors_held, vectors_kept


def _hashed(parts: Iterable[bytes], hashed: hashlib.blake2b) -> Iterator[bytes]:
    for part in parts:
        hashed.update(part)
        yield part


@dataclass(frozen=True, slots=True)
class NoteStored:
    tally: Tally
    problem: str | None = None


def _stored(
    job: Job, transaction: WritingDocuments, path: str, scanned: Identity | None, source: SourceId
) -> tuple[DocumentId, str | None, str]:
    hashed = hashlib.blake2b(digest_size=DIGEST_BYTES)
    name, title_from = posixpath.basename(path), job.collection.choices.title_from
    with job.readers.pieces(job.collection.folder / path, scanned) as parts:
        read = read_note(text_pieces(_hashed(parts, hashed)), name, title_from)
        added = add_streamed(transaction, source, path, read)
    return added, read.problem, hashed.hexdigest()


def stored_note(
    job: Job, transaction: WritingDocuments, path: str, stat: Stat, scanned: Identity | None
) -> NoteStored | Refused:
    source = put_source(transaction, path, stat.modified_ns, stat.size)
    held = kept(transaction, source)
    before = next((each for each in held if not each.key), None)
    gone = [each.id for each in held if each.key]
    delete_documents(transaction, gone)
    replaced = [before.id] if before else []
    save_point(transaction)
    vectors_held(transaction, replaced)
    delete_documents(transaction, replaced)
    try:
        added, problem, digested = _stored(job, transaction, path, scanned, source)
    except OSError as error:
        back_to_save_point(transaction)
        return Refused(error.strerror or str(error), Again.NEXT_INDEX)
    if before is not None and before.digest == digested:
        back_to_save_point(transaction)
        return NoteStored(Tally(removed=len(gone)), problem)
    set_digest(transaction, added, digested)
    vectors_kept(transaction, [added])
    past_save_point(transaction)
    return NoteStored(Tally(added=int(before is None), changed=int(before is not None), removed=len(gone)), problem)
