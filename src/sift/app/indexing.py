import posixpath
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from sift.api.errors import StoreUnavailableError
from sift.api.events import Advanced, CollectionStarted, Finished, Noted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.indexing import CollectionIndexed, IndexCollections, Indexed, IndexResult, Skipped
from sift.app.collections import chosen
from sift.app.problems import problems
from sift.domain.plan.indexing import MOST_TEXT_BYTES, IndexPlan, Stat, batches, index_plan
from sift.domain.text.notes import Document, note
from sift.ingest.scan import scan
from sift.storage.catalog import Collection
from sift.storage.database import Connection, WritingDocuments, reading
from sift.storage.documents import (
    add_documents,
    clear_unreadable,
    delete_documents,
    delete_sources,
    kept,
    mark_unreadable,
    put_source,
    sources,
    unreadable,
)
from sift.storage.keywords import writing_documents
from sift.storage.store import open_store, unavailable_if_damaged

FOLDER_MISSING_REMEDY = "connect the drive it lies on"
OUT_OF_REACH_REMEDY = "see that its folder and its store can be read, then index it again"
MOST_BYTES_A_BYTE_DECODES_TO = 3


@dataclass(frozen=True, slots=True)
class Limits:
    most_text_bytes: int = MOST_TEXT_BYTES
    most_files_a_write: int = 1000
    most_bytes_a_write: int = 64 << 20


LIMITS = Limits()


@dataclass(frozen=True, slots=True)
class Tally:
    added: int = 0
    changed: int = 0
    removed: int = 0

    def __add__(self, other: "Tally") -> "Tally":
        return Tally(self.added + other.added, self.changed + other.changed, self.removed + other.removed)


@dataclass(frozen=True, slots=True)
class Refused:
    problem: str
    read_again: bool


@dataclass(frozen=True, slots=True)
class Job:
    collection: Collection
    connection: Connection
    report: Reporter
    limits: Limits


def _chosen(home: Path, names: Sequence[str]) -> tuple[Collection, ...]:
    return chosen(home, names) if names else tuple(known for known in chosen(home, ()) if known.active)


def _too_large(size: int, limits: Limits, measured: str = "bytes") -> Refused:
    most = limits.most_text_bytes
    return Refused(f"{size:,} {measured}: a document holds at most {most:,}; split the file", read_again=False)


def _read(job: Job, path: str) -> Document | Refused:
    try:
        raw = (job.collection.folder / path).read_bytes()
    except OSError as error:
        return Refused(error.strerror or str(error), read_again=True)
    document = note(raw, posixpath.basename(path), job.collection.choices.title_from)
    most = job.limits.most_text_bytes
    if MOST_BYTES_A_BYTE_DECODES_TO * len(raw) > most and (size := len(document.text.encode())) > most:
        return _too_large(size, job.limits, "bytes as text")
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
    with writing_documents(job.connection) as transaction:
        removed = delete_sources(transaction, plan.drop)
        clear_unreadable(transaction, plan.forget)
        large = [
            _refused(transaction, path, files[path], _too_large(files[path].size, job.limits))
            for path in plan.too_large
        ]
    return sum(large, Tally(removed=removed))


def _read_and_stored(job: Job, plan: IndexPlan, files: Mapping[str, Stat]) -> Tally:
    tally, done, limits = Tally(), 0, job.limits
    for batch in batches(plan.read, files, limits.most_files_a_write, limits.most_bytes_a_write):
        read: list[tuple[str, Document | Refused]] = []
        for path in batch:
            read.append((path, _read(job, path)))
            done += 1
            job.report(Advanced(done))
        with writing_documents(job.connection) as transaction:
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
    with reading(job.connection) as transaction:
        outstanding = len(unreadable(transaction))
    job.report(Finished(tally.added, tally.changed, tally.removed, outstanding))
    return Indexed(len(files), tally.added, tally.changed, tally.removed, outstanding)


def _skipped(skipped: Collection, report: Reporter, reason: str, remedy: str | None) -> CollectionIndexed:
    report(Noted(reason, remedy))
    return CollectionIndexed(skipped.name, Skipped(reason, remedy))


def _indexed(one: Collection, report: Reporter, limits: Limits) -> CollectionIndexed:
    report(CollectionStarted(one.name, str(one.folder)))
    try:
        if not one.folder.is_dir():
            return _skipped(one, report, f"its folder is missing: {one.folder}", FOLDER_MISSING_REMEDY)
        with unavailable_if_damaged(one.store), closing(open_store(one.store, one.id)) as connection:
            return CollectionIndexed(one.name, _synced(Job(one, connection, report, limits)))
    except StoreUnavailableError as error:
        return _skipped(one, report, error.message, error.remedy)
    except OSError as error:
        return _skipped(one, report, f"it is out of reach ({error.strerror or error})", OUT_OF_REACH_REMEDY)


def index(home: Path, request: IndexCollections, report: Reporter, limits: Limits = LIMITS) -> IndexResult:
    return IndexResult(tuple(_indexed(one, report, limits) for one in _chosen(home, request.names)))
