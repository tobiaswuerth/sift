import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sift.api.errors import InvalidRequestError, StoreUnavailableError
from sift.api.events import Advanced, CollectionStarted, Finished, Noted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.indexing import CollectionIndexed, FileIndexed, IndexCollections, Indexed, IndexResult, Skipped
from sift.app.collections import chosen_or_active
from sift.app.documents import located
from sift.app.filing import LIMITS, READERS, Job, Limits, Readers, Tally
from sift.app.problems import problems
from sift.app.scans import ScanModel
from sift.app.storing import dropped, read_and_stored
from sift.domain.plan.ignore import Ignore
from sift.domain.plan.indexing import Found, Stat, dropping, index_plan, recorded
from sift.domain.text.kinds import split_keyed_path
from sift.domain.text.scans import reads_skipped
from sift.ingest.ignore import IGNORE_FILE, ignore_at
from sift.ingest.scan import Reach, Scanned, scan
from sift.storage.catalog import Collection, known_collections
from sift.storage.database import Batches, Connection, reading, writing
from sift.storage.documents import (
    Unreadable,
    holds_documents,
    source_at,
    sources,
    unreadable,
    unreadable_at,
)
from sift.storage.kept_vectors import vectors_let_go
from sift.storage.keyword_order import order_due, ordered
from sift.storage.keywords import remade, remake_started, settle
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
    tally = dropped(job, plan, files) + read_and_stored(job, plan, files, scanned.identities)
    with writing(job.connection) as transaction:
        vectors_let_go(transaction)
    job.report(PhaseDone(len(plan.read)))
    _settled(job.connection, job.report, job.limits.batches)
    with reading(job.connection) as transaction:
        outstanding = len(unreadable(transaction))
    job.report(Finished(tally.added, tally.changed, tally.removed, outstanding))
    return Indexed(len(files), tally.added, tally.changed, tally.removed, outstanding)


def _why_left_out(scanned: Scanned, path: str) -> str:
    if path in scanned.found.outside:
        return "it leads outside its collection's folder"
    return "its collection's rules leave it out" if scanned.found.left_out else "sift takes no such file there"


def _held_as_it_is(job: Job, path: str) -> tuple[bool, str | None]:
    with reading(job.connection) as transaction:
        entry, source = unreadable_at(transaction, path), source_at(transaction, path)
        if entry is None:
            return True, None
        stored = source is not None and holds_documents(transaction, source.id)
    return stored and not entry.read_again and entry.stopped_under is None, entry.problem


def _file_indexed(job: Job, ignore: Ignore, path: str) -> FileIndexed:
    shown, folder = f"{job.collection.name}/{path}", job.collection.folder
    started = job.readers.clock()
    scanned = job.readers.scan_one(folder, path, Reach(job.collection.choices.follow_links, ignore))
    files = recorded(scanned.found.files, started, job.readers.clock())
    if path in scanned.refusals:
        return FileIndexed(shown, "unreadable", scanned.refusals[path])
    with reading(job.connection) as transaction:
        held = source_at(transaction, path)
    gone = not os.path.lexists(folder / path)
    if held is None and path not in files:
        return FileIndexed(shown, "missing") if gone else FileIndexed(shown, "left out", _why_left_out(scanned, path))
    plan = index_plan(Found(files), {}, job.limits.most_file_bytes) if path in files else dropping((path,))
    tally = dropped(job, plan, files) + read_and_stored(job, plan, files, scanned.identities)
    if path not in files:
        return FileIndexed(shown, "removed", None if gone else _why_left_out(scanned, path))
    as_it_is, problem = _held_as_it_is(job, path)
    if not as_it_is:
        return FileIndexed(shown, "unreadable", problem)
    return FileIndexed(shown, "added" if held is None else "changed" if tally != Tally() else "unchanged", problem)


def _files_synced(job: Job, ignore: Ignore, paths: tuple[str, ...]) -> CollectionIndexed:
    job.report(PhaseStarted(Phase.READ, len(paths)))
    found = tuple(_file_indexed(job, ignore, path) for path in paths)
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


def _skipped(skipped: Collection, report: Reporter, reason: str, remedy: str | None) -> CollectionIndexed:
    report(Noted(reason, remedy))
    return CollectionIndexed(skipped.name, Skipped(reason, remedy))


def _ignoring(job: Job, paths: tuple[str, ...]) -> CollectionIndexed:
    try:
        ignore = ignore_at(job.collection.folder)
    except OSError as error:
        reason = f"its {IGNORE_FILE} can't be read ({error.strerror or error})"
        return _skipped(job.collection, job.report, reason, IGNORE_REMEDY)
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
    report: Reporter
    limits: Limits
    readers: Readers
    ocr: ScanModel


def _indexed(one: Collection, asked: _Indexing, paths: tuple[str, ...] = ()) -> CollectionIndexed:
    report, limits = asked.report, asked.limits
    report(CollectionStarted(one.name, str(one.folder)))
    try:
        with sole_writer(one.store, "sift index"), connected(one.store, one.id) as connection:
            _ordered(connection, report, limits.batches)
            _remade(connection, report, limits.batches)
            if not one.folder.is_dir():
                return _skipped(one, report, f"its folder is missing: {one.folder}", FOLDER_MISSING_REMEDY)
            return _ignoring(Job(one, connection, report, limits, asked.readers, asked.ocr), paths)
    except StoreUnavailableError as error:
        return _skipped(one, report, error.message, error.remedy)
    except OSError as error:
        return _skipped(one, report, f"it is out of reach ({error.strerror or error})", OUT_OF_REACH_REMEDY)


def _grouped(home: Path, files: Sequence[str]) -> dict[Collection, tuple[str, ...]]:
    known, grouped = known_collections(home), dict[Collection, list[str]]()
    for given in files:
        found = located(known, given)
        if found is None:
            raise InvalidRequestError(f"{given} is in no collection", NO_COLLECTION_REMEDY, "files")
        grouped.setdefault(found[0], []).append(split_keyed_path(found[1])[0])
    return {one: tuple(dict.fromkeys(paths)) for one, paths in grouped.items()}


def index(
    home: Path, request: IndexCollections, report: Reporter, limits: Limits = LIMITS, readers: Readers = READERS
) -> IndexResult:
    asked = _Indexing(report, limits, readers, ScanModel(readers.scanning))
    if not request.files:
        return IndexResult(tuple(_indexed(one, asked) for one in chosen_or_active(home, request.names)))
    if request.names:
        raise InvalidRequestError("give collections or files, not both", BOTH_REMEDY, "files")
    return IndexResult(tuple(_indexed(one, asked, paths) for one, paths in _grouped(home, request.files).items()))
