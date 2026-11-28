import posixpath
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path

from sift.api.collections import Retitled, TitleFrom
from sift.api.events import Advanced, CollectionStarted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.ids import DocumentId
from sift.app.filing import LIMITS, Limits
from sift.domain.errors import DamagedError
from sift.domain.text.kinds import FileKind, kind_of
from sift.domain.text.notes import NoteRead, NoteTitle, read_note
from sift.storage.catalog import Choices, Collection, choices_json, choices_of, set_choices
from sift.storage.database import Connection, WritingDocuments, reading, writing
from sift.storage.documents import add_streamed, set_digest
from sift.storage.forgetting import delete_documents
from sift.storage.kept_vectors import vectors_held, vectors_kept, vectors_let_go
from sift.storage.keywords import writing_documents
from sift.storage.retitling import TitledNote, fulfilled, promise, promised, titled_after, whole_files
from sift.storage.store import connected
from sift.storage.texts import text_of
from sift.storage.vector_meta import left
from sift.storage.writer import sole_writer


def _retitled_one(transaction: WritingDocuments, note: TitledNote, title_from: TitleFrom) -> bool:
    if kind_of(note.path) is not FileKind.NOTE:
        return False
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
        whole = whole_files(transaction)
    report(PhaseStarted(Phase.TITLE, whole.files))
    changed, done, after = 0, 0, DocumentId(0)
    while True:
        with writing_documents(connection, limits.batches) as transaction:
            batch = titled_after(transaction, after, whole.last, limits.most_files_a_write)
            changed += sum(_retitled_one(transaction, note, title_from) for note in batch)
        if not batch:
            break
        done, after = done + len(batch), batch[-1].id
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
        fulfilled(transaction)
    return Retitled(changed, after - before)


def retitle(home: Path, one: Collection, wanted: Choices, report: Reporter, limits: Limits = LIMITS) -> Retitled:
    report(CollectionStarted(one.name, str(one.folder)))
    with sole_writer(one.store, "a retitling"), connected(one.store, one.id) as connection:
        with writing(connection) as transaction:
            promise(transaction, choices_json(wanted))
        return _finished(connection, wanted.title_from, partial(set_choices, home, one.id, wanted), report, limits)


def resumed(home: Path, one: Collection, connection: Connection, report: Reporter, limits: Limits) -> Collection:
    with reading(connection) as transaction:
        owed = promised(transaction)
    if owed is None:
        return one
    try:
        wanted = choices_of(owed)
    except (ValueError, TypeError) as error:
        raise DamagedError(f"the retitling it was to finish cannot be read: {error}") from None
    _finished(connection, wanted.title_from, partial(set_choices, home, one.id, wanted), report, limits)
    return replace(one, choices=wanted)
