import itertools
import posixpath
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import NamedTuple

from sift.api.collections import Retitled, TitleFrom
from sift.api.events import Advanced, CollectionStarted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.ids import DocumentId
from sift.app.filing import LIMITS, Limits
from sift.domain.text.kinds import FileKind, kind_of
from sift.domain.text.notes import NoteRead, NoteTitle, read_note
from sift.storage.catalog import Choices, Collection, choices_json, choices_of, set_choices
from sift.storage.database import Connection, Value, WritingDocuments, first, reading, rows, run, writing
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
RETITLING = "retitling"
PROMISED = "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)"


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


def _titles_changed(connection: Connection, title_from: TitleFrom, report: Reporter, limits: Limits) -> int:
    with reading(connection) as transaction:
        notes = [each for each in rows(transaction, _Note, NOTES) if kind_of(each.path) is FileKind.NOTE]
    report(PhaseStarted(Phase.TITLE, len(notes)))
    changed = done = 0
    for batch in itertools.batched(notes, limits.most_files_a_write):
        with writing_documents(connection, limits.batches) as transaction:
            changed += sum(_retitled_one(transaction, note, title_from) for note in batch)
        done += len(batch)
        report(Advanced(done))
    report(PhaseDone(done))
    return changed


def _finished(
    connection: Connection, title_from: TitleFrom, chosen: Callable[[], None], report: Reporter, limits: Limits
) -> Retitled:
    with reading(connection) as transaction:
        before = left(transaction)
    changed = _titles_changed(connection, title_from, report, limits)
    with writing(connection) as transaction:
        vectors_let_go(transaction)
        after = left(transaction)
    chosen()
    with writing(connection) as transaction:
        run(transaction, "DELETE FROM meta WHERE key = ?", (RETITLING,))
    return Retitled(changed, after - before)


def retitle(home: Path, one: Collection, wanted: Choices, report: Reporter, limits: Limits = LIMITS) -> Retitled:
    report(CollectionStarted(one.name, str(one.folder)))
    with sole_writer(one.store, "a retitling"), connected(one.store, one.id) as connection:
        with writing(connection) as transaction:
            run(transaction, PROMISED, (RETITLING, choices_json(wanted)))
        return _finished(connection, wanted.title_from, partial(set_choices, home, one.id, wanted), report, limits)


def resumed(home: Path, one: Collection, connection: Connection, report: Reporter, limits: Limits) -> Collection:
    with reading(connection) as transaction:
        promised = first(transaction, Value, "SELECT value FROM meta WHERE key = ?", (RETITLING,))
    if promised is None:
        return one
    wanted = choices_of(promised.value)
    _finished(connection, wanted.title_from, partial(set_choices, home, one.id, wanted), report, limits)
    return replace(one, choices=wanted)
