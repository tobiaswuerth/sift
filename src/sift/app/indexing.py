import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sift.api.errors import StoreUnavailableError
from sift.api.events import Advanced, CollectionStarted, Finished, Noted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.indexing import CollectionIndexed, IndexCollections, Indexed, IndexResult, Skipped
from sift.app.collections import chosen_or_active
from sift.app.filing import READERS, Job, Limits, Readers, Tally
from sift.app.problems import problems
from sift.app.records import stored_records
from sift.domain.plan.indexing import IndexPlan, Stat, batches, index_plan, most_read, recorded
from sift.domain.text.documents import FileRead, Unread
from sift.domain.text.kinds import streamed
from sift.ingest.files import Identity
from sift.ingest.readers import read_whole
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
from sift.storage.kept_vectors import vectors_held, vectors_kept
from sift.storage.keywords import remade, remake_started, settle, writing_documents
from sift.storage.store import connected

FOLDER_MISSING_REMEDY = "connect the drive it lies on"
OUT_OF_REACH_REMEDY = "see that its folder and its store can be read, then index it again"
MOST_BYTES_A_CHARACTER_ENCODES_TO = 4
PROBLEMS_APART = "; "


LIMITS = Limits()


@dataclass(frozen=True, slots=True)
class Refused:
    problem: str
    read_again: bool


def _too_large(size: str, most: int, measured: str = "bytes") -> Refused:
    return Refused(f"{size} {measured}: a document holds at most {most:,}; split the file", read_again=False)


def _largest_text(read: FileRead, most: int) -> int | None:
    for each in read.documents:
        if MOST_BYTES_A_CHARACTER_ENCODES_TO * len(each.text) > most and (size := len(each.text.encode())) > most:
            return size
    return None


def _read(job: Job, path: str, scanned: Identity | None) -> FileRead | Refused:
    most, most_text = most_read(path, job.limits.sizes), job.limits.most_text_bytes
    try:
        raw = job.readers.whole(job.collection.folder / path, most, scanned)
    except OSError as error:
        return Refused(error.strerror or str(error), read_again=True)
    if len(raw) > most:
        return _too_large(f"more than {most:,}", most)
    read = read_whole(raw, posixpath.basename(path), job.collection.choices.title_from, job.readers.convert)
    if isinstance(read, Unread):
        return Refused(read.problem, read_again=False)
    size = _largest_text(read, most_text)
    return read if size is None else _too_large(f"{size:,}", most_text, "bytes as text")


def _refused(transaction: WritingDocuments, path: str, stat: Stat, refused: Refused) -> Tally:
    mark_unreadable(transaction, path, refused.problem, read_again=refused.read_again)
    if refused.read_again:
        return Tally()
    source = put_source(transaction, path, stat.modified_ns, stat.size)
    gone = [document.id for document in kept(transaction, source)]
    delete_documents(transaction, gone)
    return Tally(removed=len(gone))


def _stored(transaction: WritingDocuments, path: str, stat: Stat, read: FileRead | Refused) -> Tally:
    match read:
        case Refused():
            return _refused(transaction, path, stat, read)
        case _:
            return _file_stored(transaction, path, stat, read)


def _file_stored(transaction: WritingDocuments, path: str, stat: Stat, read: FileRead) -> Tally:
    source = put_source(transaction, path, stat.modified_ns, stat.size)
    held = {document.key: document for document in kept(transaction, source)}
    changed = [each for each in read.documents if each.key in held and held[each.key].digest != each.digest]
    new = [each for each in read.documents if each.key not in held]
    gone = [held[key].id for key in held.keys() - {each.key for each in read.documents}]
    replaced = [held[each.key].id for each in changed]
    vectors_held(transaction, replaced)
    delete_documents(transaction, [*gone, *replaced])
    vectors_kept(transaction, add_documents(transaction, source, path, [*changed, *new]))
    if read.problems:
        mark_unreadable(transaction, path, PROBLEMS_APART.join(read.problems), read_again=False)
    return Tally(added=len(new), changed=len(changed), removed=len(gone))


def _dropped(job: Job, plan: IndexPlan, files: Mapping[str, Stat]) -> Tally:
    with writing_documents(job.connection, job.limits.batches) as transaction:
        removed = delete_sources(transaction, plan.drop)
        clear_unreadable(transaction, plan.forget)
        large = [
            _refused(
                transaction, path, files[path], _too_large(f"{files[path].size:,}", most_read(path, job.limits.sizes))
            )
            for path in plan.too_large
        ]
    return sum(large, Tally(removed=removed))


def _read_and_stored(job: Job, plan: IndexPlan, files: Mapping[str, Stat], identities: Mapping[str, Identity]) -> Tally:
    tally, done, limits = Tally(), 0, job.limits
    whole = tuple(path for path in plan.read if not streamed(path))
    for path in (path for path in plan.read if streamed(path)):
        tally += stored_records(job, path, files[path], job.collection.choices.records, identities.get(path))
        done += 1
        job.report(Advanced(done))
    for batch in batches(whole, files, limits.most_files_a_write, limits.most_bytes_a_write):
        read: list[tuple[str, FileRead | Refused]] = []
        for path in batch:
            read.append((path, _read(job, path, identities.get(path))))
            done += 1
            job.report(Advanced(done))
        with writing_documents(job.connection, job.limits.batches) as transaction:
            for path, outcome in read:
                tally += _stored(transaction, path, files[path], outcome)
            clear_unreadable(transaction, [path for path, outcome in read if _whole(outcome)])
        for path, outcome in read:
            if isinstance(outcome, FileRead) and outcome.problems:
                job.report(Noted(f"{path}: {PROBLEMS_APART.join(outcome.problems)}"))
    return tally


def _whole(outcome: FileRead | Refused) -> bool:
    return isinstance(outcome, FileRead) and not outcome.problems


def _synced(job: Job) -> Indexed:
    job.report(PhaseStarted(Phase.SCAN, None))
    started = job.readers.clock()
    scanned = scan(
        job.collection.folder,
        follow_links=job.collection.choices.follow_links,
        progress=lambda found: job.report(Advanced(found)),
    )
    files = recorded(scanned.found.files, started)
    job.report(PhaseDone(len(files)))
    with reading(job.connection) as transaction:
        known = {path: Stat(source.modified_ns, source.size) for path, source in sources(transaction).items()}
        marks = {entry.path: bool(entry.read_again) for entry in unreadable(transaction)}
    plan = index_plan(scanned.found, known, job.limits.sizes, marks)
    for said in problems(job.collection.folder, scanned, plan, frozenset(known)):
        job.report(said)
    job.report(PhaseStarted(Phase.READ, len(plan.read)))
    tally = _dropped(job, plan, files) + _read_and_stored(job, plan, files, scanned.identities)
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
        with connected(one.store, one.id) as connection:
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
