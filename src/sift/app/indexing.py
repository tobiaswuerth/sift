import posixpath
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from sift.api.errors import StoreUnavailableError
from sift.api.events import Advanced, CollectionStarted, Finished, Noted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.indexing import CollectionIndexed, IndexCollections, Indexed, IndexResult, Skipped
from sift.app.collections import chosen_or_active
from sift.app.filing import READERS, Job, Limits, Readers, Tally
from sift.app.problems import problems
from sift.app.records import stored_records
from sift.domain.plan.indexing import IndexPlan, Stat, batches, index_plan
from sift.domain.text.notes import Document, note
from sift.domain.text.records import RECORD_FILES
from sift.ingest.scan import scan
from sift.storage.catalog import Collection
from sift.storage.database import Batches, Connection, WritingDocuments, reading
from sift.storage.documents import (
    add_documents,
    clear_unreadable,
    kept,
    mark_unreadable,
    put_source,
    sources,
    unreadable,
)
from sift.storage.forgetting import delete_documents, delete_sources
from sift.storage.keywords import remade, remake_started, settle, writing_documents
from sift.storage.store import open_store, unavailable_if_unusable

FOLDER_MISSING_REMEDY = "connect the drive it lies on"
OUT_OF_REACH_REMEDY = "see that its folder and its store can be read, then index it again"
MOST_BYTES_A_BYTE_DECODES_TO = 3


LIMITS = Limits()


@dataclass(frozen=True, slots=True)
class Refused:
    problem: str
    read_again: bool


def _too_large(size: str, limits: Limits, measured: str = "bytes") -> Refused:
    most = limits.most_text_bytes
    return Refused(f"{size} {measured}: a document holds at most {most:,}; split the file", read_again=False)


def _read(job: Job, path: str) -> Document | Refused:
    most = job.limits.most_text_bytes
    try:
        raw = job.readers.whole(job.collection.folder / path, most)
    except OSError as error:
        return Refused(error.strerror or str(error), read_again=True)
    if len(raw) > most:
        return _too_large(f"more than {most:,}", job.limits)
    document = note(raw, posixpath.basename(path), job.collection.choices.title_from)
    if MOST_BYTES_A_BYTE_DECODES_TO * len(raw) > most and (size := len(document.text.encode())) > most:
        return _too_large(f"{size:,}", job.limits, "bytes as text")
    return document


def _refused(transaction: WritingDocuments, path: str, stat: Stat, refused: Refused) -> Tally:
    mark_unreadable(transaction, path, refused.problem, read_again=refused.read_again)
    if refused.read_again:
        return Tally()
    source = put_source(transaction, path, stat.modified_ns, stat.size)
    gone = [document.id for document in kept(transaction, source)]
    delete_documents(transaction, gone)
    return Tally(removed=len(gone))


def _stored(transaction: WritingDocuments, path: str, stat: Stat, read: Document | Refused) -> Tally:
    if isinstance(read, Refused):
        return _refused(transaction, path, stat, read)
    source = put_source(transaction, path, stat.modified_ns, stat.size)
    held = {document.key: document for document in kept(transaction, source)}
    before = held.pop(read.key, None)
    gone = [document.id for document in held.values()]
    if before is not None and before.digest == read.digest:
        delete_documents(transaction, gone)
        return Tally(removed=len(gone))
    delete_documents(transaction, [*gone, *([before.id] if before else [])])
    add_documents(transaction, source, path, [read])
    return Tally(added=int(before is None), changed=int(before is not None), removed=len(gone))


def _dropped(job: Job, plan: IndexPlan, files: Mapping[str, Stat]) -> Tally:
    with writing_documents(job.connection, job.limits.batches) as transaction:
        removed = delete_sources(transaction, plan.drop)
        clear_unreadable(transaction, plan.forget)
        large = [
            _refused(transaction, path, files[path], _too_large(f"{files[path].size:,}", job.limits))
            for path in plan.too_large
        ]
    return sum(large, Tally(removed=removed))


def _records_stored(job: Job, path: str, stat: Stat) -> Tally:
    try:
        return stored_records(job, path, stat, job.collection.choices.records)
    except OSError as error:
        with writing_documents(job.connection, job.limits.batches) as transaction:
            return _refused(transaction, path, stat, Refused(error.strerror or str(error), read_again=True))


def _read_and_stored(job: Job, plan: IndexPlan, files: Mapping[str, Stat]) -> Tally:
    tally, done, limits = Tally(), 0, job.limits
    whole = tuple(path for path in plan.read if not path.lower().endswith(RECORD_FILES))
    for path in (path for path in plan.read if path.lower().endswith(RECORD_FILES)):
        tally += _records_stored(job, path, files[path])
        done += 1
        job.report(Advanced(done))
    for batch in batches(whole, files, limits.most_files_a_write, limits.most_bytes_a_write):
        read: list[tuple[str, Document | Refused]] = []
        for path in batch:
            read.append((path, _read(job, path)))
            done += 1
            job.report(Advanced(done))
        with writing_documents(job.connection, job.limits.batches) as transaction:
            for path, outcome in read:
                tally += _stored(transaction, path, files[path], outcome)
            clear_unreadable(transaction, [path for path, outcome in read if isinstance(outcome, Document)])
    return tally


def _synced(job: Job) -> Indexed:
    job.report(PhaseStarted(Phase.SCAN, None))
    scanned = scan(
        job.collection.folder,
        follow_links=job.collection.choices.follow_links,
        progress=lambda found: job.report(Advanced(found)),
    )
    files = scanned.found.files
    job.report(PhaseDone(len(files)))
    with reading(job.connection) as transaction:
        known = {path: Stat(source.modified_ns, source.size) for path, source in sources(transaction).items()}
        marks = {entry.path: bool(entry.read_again) for entry in unreadable(transaction)}
    plan = index_plan(scanned.found, known, job.limits.most_text_bytes, marks)
    for said in problems(job.collection.folder, scanned, plan, frozenset(known)):
        job.report(said)
    job.report(PhaseStarted(Phase.READ, len(plan.read)))
    tally = _dropped(job, plan, files) + _read_and_stored(job, plan, files)
    job.report(PhaseDone(len(plan.read)))
    _settled(job.connection, job.report, job.limits.batches)
    with reading(job.connection) as transaction:
        outstanding = len(unreadable(transaction))
    job.report(Finished(tally.added, tally.changed, tally.removed, outstanding))
    return Indexed(len(files), tally.added, tally.changed, tally.removed, outstanding)


def _skipped(skipped: Collection, report: Reporter, reason: str, remedy: str | None) -> CollectionIndexed:
    report(Noted(reason, remedy))
    return CollectionIndexed(skipped.name, Skipped(reason, remedy))


def _remade(connection: Connection, report: Reporter, batches: Batches) -> None:
    left = remake_started(connection)
    if not left:
        return
    report(PhaseStarted(Phase.REMAKE, left))
    done = 0
    for done in remade(connection, batches):
        report(Advanced(done))
    report(PhaseDone(done))
    _settled(connection, report, batches)


def _settled(connection: Connection, report: Reporter, batches: Batches) -> None:
    done, started = 0, False
    for done in settle(connection, batches):
        report(Advanced(done) if started else PhaseStarted(Phase.MERGE, None))
        started = True
    if started:
        report(PhaseDone(done))


def _indexed(one: Collection, report: Reporter, limits: Limits, readers: Readers) -> CollectionIndexed:
    report(CollectionStarted(one.name, str(one.folder)))
    try:
        with unavailable_if_unusable(one.store), closing(open_store(one.store, one.id)) as connection:
            _remade(connection, report, limits.batches)
            if not one.folder.is_dir():
                return _skipped(one, report, f"its folder is missing: {one.folder}", FOLDER_MISSING_REMEDY)
            return CollectionIndexed(one.name, _synced(Job(one, connection, report, limits, readers)))
    except StoreUnavailableError as error:
        return _skipped(one, report, error.message, error.remedy)
    except OSError as error:
        return _skipped(one, report, f"it is out of reach ({error.strerror or error})", OUT_OF_REACH_REMEDY)


def index(
    home: Path, request: IndexCollections, report: Reporter, limits: Limits = LIMITS, readers: Readers = READERS
) -> IndexResult:
    chosen = chosen_or_active(home, request.names)
    return IndexResult(tuple(_indexed(one, report, limits, readers) for one in chosen))
