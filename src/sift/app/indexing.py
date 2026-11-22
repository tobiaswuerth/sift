import posixpath
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from sift.api.errors import StoreUnavailableError
from sift.api.events import Advanced, CollectionStarted, Finished, Noted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.indexing import CollectionIndexed, IndexCollections, Indexed, IndexResult, Skipped
from sift.app.problems import problems
from sift.domain.plan.indexing import MOST_TEXT_BYTES, IndexPlan, Stat, batches, index_plan
from sift.domain.text.notes import Document, note
from sift.ingest.scan import scan
from sift.storage.catalog import Collection, collection, collections, open_catalog
from sift.storage.database import Connection, Writing, reading, writing
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
from sift.storage.store import open_store

FOLDER_MISSING_REMEDY = "connect the drive it lies on"


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
    unreadable: int = 0

    def __add__(self, other: "Tally") -> "Tally":
        return Tally(
            self.added + other.added,
            self.changed + other.changed,
            self.removed + other.removed,
            self.unreadable + other.unreadable,
        )


@dataclass(frozen=True, slots=True)
class Job:
    collection: Collection
    connection: Connection
    report: Reporter
    limits: Limits


def _chosen(home: Path, names: Sequence[str]) -> tuple[Collection, ...]:
    with closing(open_catalog(home)) as catalog, reading(catalog) as transaction:
        if not names:
            return tuple(known for known in collections(transaction) if known.active)
        return tuple(dict.fromkeys(collection(transaction, name) for name in names))


def _read(job: Job, path: str) -> Document | str:
    try:
        raw = (job.collection.folder / path).read_bytes()
    except OSError as error:
        return error.strerror or str(error)
    return note(raw, posixpath.basename(path), job.collection.choices.title_from)


def _stored(transaction: Writing, path: str, stat: Stat, read: Document | str) -> Tally:
    if isinstance(read, str):
        mark_unreadable(transaction, path, read)
        return Tally(unreadable=1)
    source = put_source(transaction, path, stat.modified_ns, stat.size)
    held = {document.key: document for document in kept(transaction, source)}
    before = held.pop(read.key, None)
    gone = [document.id for document in held.values()]
    clear_unreadable(transaction, [path])
    if before is not None and before.digest == read.digest:
        delete_documents(transaction, gone)
        return Tally(removed=len(gone))
    delete_documents(transaction, [*gone, *([before.id] if before else [])])
    add_documents(transaction, source, path, [read])
    return Tally(added=int(before is None), changed=int(before is not None), removed=len(gone))


def _too_large(transaction: Writing, path: str, stat: Stat, limits: Limits) -> Tally:
    source = put_source(transaction, path, stat.modified_ns, stat.size)
    gone = [document.id for document in kept(transaction, source)]
    delete_documents(transaction, gone)
    most = limits.most_text_bytes
    mark_unreadable(transaction, path, f"{stat.size:,} bytes: a document holds at most {most:,}; split the file")
    return Tally(removed=len(gone), unreadable=1)


def _dropped(job: Job, plan: IndexPlan, stale: Sequence[str], files: Mapping[str, Stat]) -> Tally:
    with writing(job.connection) as transaction:
        removed = delete_sources(transaction, plan.drop)
        clear_unreadable(transaction, [*plan.drop, *stale])
        large = [_too_large(transaction, path, files[path], job.limits) for path in plan.too_large]
    return sum(large, Tally(removed=removed))


def _read_and_stored(job: Job, plan: IndexPlan, files: Mapping[str, Stat]) -> Tally:
    tally, done, limits = Tally(), 0, job.limits
    for batch in batches(plan.read, files, limits.most_files_a_write, limits.most_bytes_a_write):
        read = [(path, _read(job, path)) for path in batch]
        with writing(job.connection) as transaction:
            for path, outcome in read:
                tally += _stored(transaction, path, files[path], outcome)
                done += 1
                job.report(Advanced(done))
    return tally


def _synced(job: Job) -> Indexed:
    job.report(PhaseStarted(Phase.SCAN, None))
    scanned = scan(job.collection.folder, follow_links=job.collection.choices.follow_links)
    files = scanned.found.files
    job.report(PhaseDone(len(files)))
    with reading(job.connection) as transaction:
        known = {path: Stat(source.modified_ns, source.size) for path, source in sources(transaction).items()}
        listed = [entry.path for entry in unreadable(transaction)]
    plan = index_plan(scanned.found, known, job.limits.most_text_bytes)
    for said in problems(job.collection.folder, scanned, plan, frozenset(known)):
        job.report(said)
    stale = [path for path in listed if path not in files and path not in plan.kept]
    job.report(PhaseStarted(Phase.READ, len(plan.read)))
    tally = _dropped(job, plan, stale, files) + _read_and_stored(job, plan, files)
    job.report(PhaseDone(len(plan.read)))
    job.report(Finished(tally.added, tally.changed, tally.removed, tally.unreadable))
    return Indexed(len(files), tally.added, tally.changed, tally.removed, tally.unreadable)


def _skipped(chosen: Collection, report: Reporter, reason: str, remedy: str | None) -> CollectionIndexed:
    report(Noted(reason, remedy))
    return CollectionIndexed(chosen.name, Skipped(reason, remedy))


def _indexed(chosen: Collection, report: Reporter, limits: Limits) -> CollectionIndexed:
    report(CollectionStarted(chosen.name, str(chosen.folder)))
    if not chosen.folder.is_dir():
        return _skipped(chosen, report, f"its folder is missing: {chosen.folder}", FOLDER_MISSING_REMEDY)
    try:
        connection = open_store(chosen.store, chosen.id)
    except StoreUnavailableError as error:
        return _skipped(chosen, report, error.message, error.remedy)
    with closing(connection):
        return CollectionIndexed(chosen.name, _synced(Job(chosen, connection, report, limits)))


def index(home: Path, request: IndexCollections, report: Reporter, limits: Limits = LIMITS) -> IndexResult:
    return IndexResult(tuple(_indexed(chosen, report, limits) for chosen in _chosen(home, request.names)))
