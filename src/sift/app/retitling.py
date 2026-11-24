import posixpath
from collections.abc import Sequence
from typing import NamedTuple

from sift.api.collections import TitleFrom
from sift.api.events import Advanced, CollectionStarted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.ids import DocumentId
from sift.app.choosing import Retitled
from sift.app.filing import LIMITS, Limits
from sift.domain.text.kinds import FileKind, kind_of
from sift.domain.text.notes import NoteRead, NoteTitle, read_note
from sift.storage.catalog import Collection
from sift.storage.database import WritingDocuments, reading, rows, writing
from sift.storage.documents import SourceId, add_streamed, set_digest
from sift.storage.forgetting import delete_documents
from sift.storage.kept_vectors import vectors_held, vectors_kept, vectors_let_go
from sift.storage.keywords import writing_documents
from sift.storage.store import connected
from sift.storage.texts import text_of
from sift.storage.vector_meta import left
from sift.storage.writer import sole_writer

NOTES = """SELECT d.id, d.source, s.path, d.title, d.digest FROM documents d JOIN sources s ON s.id = d.source
           WHERE d.key = '' ORDER BY d.id"""
AT_ONCE = 500


class _Note(NamedTuple):
    id: DocumentId
    source: SourceId
    path: str
    title: str
    digest: str


def _retitled_one(transaction: WritingDocuments, note: _Note, title_from: TitleFrom) -> bool:
    read = read_note((text_of(transaction, note.id),), posixpath.basename(note.path), title_from)
    parts = list(read.parts)
    title = next(part.title for part in reversed(parts) if isinstance(part, NoteTitle))
    if title == note.title:
        return False
    vectors_held(transaction, [note.id])
    delete_documents(transaction, [note.id])
    added = add_streamed(transaction, note.source, note.path, NoteRead(read.fields, iter(parts)))
    set_digest(transaction, added, note.digest)
    vectors_kept(transaction, [added])
    return True


def _batches(notes: Sequence[_Note]) -> list[Sequence[_Note]]:
    return [notes[at : at + AT_ONCE] for at in range(0, len(notes), AT_ONCE)]


def retitle(one: Collection, title_from: TitleFrom, report: Reporter, limits: Limits = LIMITS) -> Retitled:
    report(CollectionStarted(one.name, str(one.folder)))
    with sole_writer(one.store, "sift add"), connected(one.store, one.id) as connection:
        with reading(connection) as transaction:
            before = left(transaction)
            notes = [each for each in rows(transaction, _Note, NOTES) if kind_of(each.path) is FileKind.NOTE]
        report(PhaseStarted(Phase.TITLE, len(notes)))
        changed = done = 0
        for batch in _batches(notes):
            with writing_documents(connection, limits.batches) as transaction:
                changed += sum(_retitled_one(transaction, note, title_from) for note in batch)
            done += len(batch)
            report(Advanced(done))
        with writing(connection) as transaction:
            vectors_let_go(transaction)
            after = left(transaction)
    report(PhaseDone(done))
    return Retitled(changed, after - before)
