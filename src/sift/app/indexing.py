import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from sift.api.collections import Audience
from sift.api.errors import InvalidRequestError, Problem, StoreUnavailableError
from sift.api.events import (
    Advanced,
    CollectionStarted,
    Event,
    FilesKept,
    Finished,
    Noted,
    Phase,
    PhaseDone,
    PhaseStarted,
    Reporter,
)
from sift.api.indexing import CollectionIndexed, FileIndexed, IndexCollections, Indexed, IndexResult, Skipped
from sift.app.collections import chosen_or_active
from sift.app.documents import located
from sift.app.filing import LIMITS, READERS, Job, Limits, Readers, Tally
from sift.app.problems import problems
from sift.app.retitling import resumed
from sift.app.scans import ScanModel
from sift.app.storing import dropped, read_and_stored, readied
from sift.domain.plan.changes import FileOutcome, FileSeen, state_read, state_unread
from sift.domain.plan.ignore import Ignore
from sift.domain.plan.indexing import Found, Stat, dropping, index_plan, recorded
from sift.domain.text.kinds import split_keyed_path
from sift.domain.text.scans import reads_skipped
from sift.ingest.files import Listings
from sift.ingest.ignore import IGNORE_FILE, ignore_at
from sift.ingest.scan import Reach, scan
from sift.storage.catalog import Collection, known_to
from sift.storage.database import Batches, Connection, reading, writing
from sift.storage.documents import (
    Unreadable,
    holds_documents,
    source_at,
    sources,
    unreadable,
    unreadable_at,
)
from sift.storage.forgetting import documents_at
from sift.storage.kept_vectors import vectors_held, vectors_let_go
from sift.storage.keyword_order import order_due, ordered
from sift.storage.keywords import remade, remake_started, settle
from sift.storage.leftovers import tidied
from sift.storage.store import connected
from sift.storage.writer import sole_writer

FOLDER_MISSING_REMEDY = "connect the drive it lies on"
OUT_OF_REACH_REMEDY = "see that its folder and its store can be read, then index it again"
IGNORE_REMEDY = f"make its {IGNORE_FILE} a file sift can read, or remove it, then index it again"
NO_COLLECTION_REMEDY = "give a file inside a collection's folder, or its id as collection/path"
BOTH_REMEDY = "give the collections to index, or the files to read now"


def _read_again(readers: Readers, entry: Unreadable) -> bool:
    stopped_before = entry.stopped_under is not None and entry.stopped_under != readers.release()
    return bool(entry.read_again) or reads_skipped(readers.scanning.setting, entry.skipped_under) or stopped_before


def _synced(job: Job, ignore: Ignore) -> Indexed:
    job.report(PhaseStarted(Phase.SCAN, None))
    started = job.readers.clock()
    scanned = scan(
        job.collection.folder,
        Reach(job.collection.choices.follow_links, ignore),
        progress=lambda found: job.report(Advanced(found)),
    )
    files = recorded(scanned.found.files, started, job.readers.clock())
    job.report(PhaseDone(len(files)))
    with reading(job.connection) as transaction:
        known = {path: Stat(source.modified_ns, source.size) for path, source in sources(transaction).items()}
        marks = {entry.path: _read_again(job.readers, entry) for entry in unreadable(transaction)}
    plan = index_plan(scanned.found, known, job.limits.most_file_bytes, marks, ignore)
    for said in problems(job.collection.folder, scanned, plan, frozenset(known)):
        job.report(said)
    job.report(PhaseStarted(Phase.READ, len(plan.read)))
    tally = readied(job, plan, files) + read_and_stored(job, plan, files, scanned.identities) + dropped(job, plan)
    with writing(job.connection) as transaction:
        vectors_let_go(transaction)
    job.report(PhaseDone(len(plan.read)))
    _settled(job.connection, job.report, job.limits.batches)
    with reading(job.connection) as transaction:
        outstanding = len(unreadable(transaction))
    job.report(Finished(tally.added, tally.changed, tally.removed, outstanding))
    return Indexed(len(files), tally.added, tally.changed, tally.removed, outstanding)


def _held_as_it_is(job: Job, path: str) -> tuple[bool, str | None]:
    with reading(job.connection) as transaction:
        entry, source = unreadable_at(transaction, path), source_at(transaction, path)
        if entry is None:
            return True, None
        stored = source is not None and holds_documents(transaction, source.id)
    return stored and not entry.read_again and entry.stopped_under is None, entry.problem


def _file_said(shown: str, outcome: FileOutcome) -> FileIndexed:
    return FileIndexed(shown, outcome.state, outcome.note, Problem.OS if outcome.system else None)


def _file_indexed(job: Job, ignore: Ignore, path: str) -> FileIndexed:
    shown, folder = f"{job.collection.name}/{path}", job.collection.folder
    started = job.readers.clock()
    scanned = job.readers.scan_one(folder, path, Reach(job.collection.choices.follow_links, ignore))
    files = recorded(scanned.found.files, started, job.readers.clock())
    with reading(job.connection) as transaction:
        held = source_at(transaction, path) is not None
    gone, listed = not os.path.lexists(folder / path), path in files
    seen = FileSeen(scanned.refusals.get(path), held, listed, gone, scanned.why.get(path))
    unread = state_unread(seen)
    if unread is not None:
        return _file_said(shown, unread)
    plan = index_plan(Found(files), {}, job.limits.most_file_bytes) if listed else dropping((path,))
    tally = readied(job, plan, files) + read_and_stored(job, plan, files, scanned.identities) + dropped(job, plan)
    as_it_is, problem = _held_as_it_is(job, path) if listed else (True, None)
    return _file_said(shown, state_read(seen, held_as_it_is=as_it_is, problem=problem, changed=tally != Tally()))


def _uncounted(report: Reporter) -> Reporter:
    def told(event: Event) -> None:
        if not isinstance(event, Advanced | FilesKept):
            report(event)

    return told


def _files_indexed(job: Job, ignore: Ignore, paths: tuple[str, ...]) -> tuple[FileIndexed, ...]:
    gone = frozenset(path for path in paths if not os.path.lexists(job.collection.folder / path))
    with writing(job.connection) as transaction:
        vectors_held(transaction, documents_at(transaction, sorted(gone)))
    each, found = replace(job, report=_uncounted(job.report)), dict[str, FileIndexed]()
    for path in sorted(paths, key=gone.__contains__):
        found[path] = _file_indexed(each, ignore, path)
        job.report(Advanced(len(found)))
        job.report(FilesKept(len(found)))
    return tuple(found[path] for path in paths)


def _files_synced(job: Job, ignore: Ignore, paths: tuple[str, ...]) -> CollectionIndexed:
    job.report(PhaseStarted(Phase.READ, len(paths)))
    found = _files_indexed(job, ignore, paths)
    with writing(job.connection) as transaction:
        vectors_let_go(transaction)
    job.report(PhaseDone(len(paths)))
    _settled(job.connection, job.report, job.limits.batches)
    with reading(job.connection) as transaction:
        outstanding = len(unreadable(transaction))
    states = Counter(each.state for each in found)
    job.report(Finished(states["added"], states["changed"], states["removed"], outstanding))
    outcome = Indexed(len(paths), states["added"], states["changed"], states["removed"], outstanding)
    return CollectionIndexed(job.collection.name, outcome, found)


def _skipped(skipped: Collection, report: Reporter, outcome: Skipped) -> CollectionIndexed:
    report(Noted(outcome.message, outcome.remedy, outcome.code))
    return CollectionIndexed(skipped.name, outcome)


def _ignoring(job: Job, paths: tuple[str, ...]) -> CollectionIndexed:
    try:
        ignore = ignore_at(job.collection.folder)
    except OSError as error:
        reason = f"its {IGNORE_FILE} can't be read ({error.strerror or error})"
        return _skipped(job.collection, job.report, Skipped(Problem.OS, reason, IGNORE_REMEDY))
    if paths:
        return _files_synced(job, ignore, paths)
    return CollectionIndexed(job.collection.name, _synced(job, ignore))


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


def _ordered(connection: Connection, report: Reporter, batches: Batches) -> None:
    with reading(connection) as transaction:
        due = order_due(transaction)
    if not due:
        return
    report(PhaseStarted(Phase.ORDER, None))
    report(PhaseDone(ordered(connection)))
    _remade(connection, report, batches)


def _settled(connection: Connection, report: Reporter, batches: Batches) -> None:
    done, started = 0, False
    for done in settle(connection, batches):
        report(Advanced(done) if started else PhaseStarted(Phase.MERGE, None))
        started = True
    if started:
        report(PhaseDone(done))


@dataclass(frozen=True, slots=True)
class _Indexing:
    home: Path
    report: Reporter
    limits: Limits
    readers: Readers
    ocr: ScanModel


def _indexed(one: Collection, asked: _Indexing, paths: tuple[str, ...] = ()) -> CollectionIndexed:
    report, limits = asked.report, asked.limits
    report(CollectionStarted(one.name, str(one.folder)))
    try:
        with sole_writer(one.store, "an index"), connected(one.store, one.id) as connection:
            tidied(asked.home, one.id)
            current = resumed(asked.home, one, connection, report, limits)
            _ordered(connection, report, limits.batches)
            _remade(connection, report, limits.batches)
            if not current.folder.is_dir():
                missing = Skipped(
                    Problem.FOLDER_MISSING, f"its folder is missing: {current.folder}", FOLDER_MISSING_REMEDY
                )
                return _skipped(current, report, missing)
            return _ignoring(Job(current, connection, report, limits, asked.readers, asked.ocr), paths)
    except StoreUnavailableError as error:
        return _skipped(one, report, Skipped(error.code, error.message, error.remedy))
    except OSError as error:
        reason = f"it is out of reach ({error.strerror or error})"
        return _skipped(one, report, Skipped(Problem.OS, reason, OUT_OF_REACH_REMEDY))


def _file_named(one: Collection, relative: str, listings: Listings) -> str:
    path = relative if os.path.lexists(one.folder / relative) else split_keyed_path(relative)[0]
    return listings.listed_as(one.folder, path)


def files_of(home: Path, files: Sequence[str], audience: Audience) -> Mapping[Collection, tuple[str, ...]]:
    known, grouped, listings = known_to(home, audience), dict[Collection, list[str]](), Listings()
    for given in files:
        found = located(known, given)
        if found is None:
            raise InvalidRequestError(f"{given} is in no collection", NO_COLLECTION_REMEDY, "files")
        grouped.setdefault(found[0], []).append(_file_named(*found, listings))
    return MappingProxyType({one: tuple(dict.fromkeys(paths)) for one, paths in grouped.items()})


def index_files(
    home: Path,
    grouped: Mapping[Collection, tuple[str, ...]],
    report: Reporter,
    limits: Limits = LIMITS,
    readers: Readers = READERS,
) -> IndexResult:
    asked = _Indexing(home, report, limits, readers, ScanModel(readers.scanning))
    try:
        return IndexResult(tuple(_indexed(one, asked, paths) for one, paths in grouped.items()))
    finally:
        asked.ocr.let_go()


def index(
    home: Path, request: IndexCollections, report: Reporter, limits: Limits = LIMITS, readers: Readers = READERS
) -> IndexResult:
    if request.files and request.names:
        raise InvalidRequestError("give collections or files, not both", BOTH_REMEDY, "files")
    if request.files:
        return index_files(home, files_of(home, request.files, Audience.PEOPLE), report, limits, readers)
    asked = _Indexing(home, report, limits, readers, ScanModel(readers.scanning))
    try:
        return IndexResult(
            tuple(_indexed(one, asked) for one in chosen_or_active(home, request.names, Audience.PEOPLE))
        )
    finally:
        asked.ocr.let_go()
