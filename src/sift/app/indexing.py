from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from sift.api.errors import StoreUnavailableError
from sift.api.events import Advanced, CollectionStarted, Finished, Noted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.indexing import CollectionIndexed, IndexCollections, Indexed, IndexResult, Skipped
from sift.app.collections import chosen_or_active
from sift.app.filing import READERS, Job, Limits, Readers, Tally, too_large_to_describe
from sift.app.problems import problems
from sift.app.records import stored_records
from sift.app.streamed import stored_note
from sift.domain.plan.ignore import Ignore
from sift.domain.plan.indexing import IndexPlan, Stat, batches, index_plan, recorded
from sift.domain.text.documents import FileRead, Outcome, Refused, too_large
from sift.domain.text.kinds import FileKind, kind_of, streamed
from sift.ingest.files import Identity
from sift.ingest.ignore import IGNORE_FILE, ignore_at
from sift.ingest.readers import ToRead
from sift.ingest.sandbox import FileReading
from sift.ingest.scan import Reach, scan
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
IGNORE_REMEDY = f"make its {IGNORE_FILE} a file sift can read, or remove it, then index it again"
PROBLEMS_APART = "; "


LIMITS = Limits()


def _described(read: FileRead, limits: Limits) -> FileRead:
    found = [(each, too_large_to_describe(each, limits.most_description_bytes)) for each in read.documents]
    kept = tuple(each for each, problem in found if problem is None)
    large = (f"{each.key}: {problem}" if each.key else problem for each, problem in found if problem is not None)
    return FileRead(kept, (*read.problems, *large))


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
    most = job.limits.most_file_bytes
    with writing_documents(job.connection, job.limits.batches) as transaction:
        removed = delete_sources(transaction, plan.drop)
        clear_unreadable(transaction, plan.forget)
        large = [_refused(transaction, path, files[path], too_large(files[path].size, most)) for path in plan.too_large]
    return sum(large, Tally(removed=removed))


@dataclass(slots=True)
class _Run:
    job: Job
    files: Mapping[str, Stat]
    identities: Mapping[str, Identity]
    done: int = 0

    def advanced(self) -> None:
        self.done += 1
        self.job.report(Advanced(self.done))

    def asked(self, path: str) -> ToRead:
        job = self.job
        most, title_from = job.limits.most_file_bytes, job.collection.choices.title_from
        return ToRead(job.collection.folder / path, self.files[path].size, most, self.identities.get(path), title_from)


def _read(run: _Run, reading: FileReading, paths: Sequence[str]) -> list[tuple[str, Outcome]]:
    outcomes: dict[int, Outcome] = {}
    for at, outcome in reading([run.asked(path) for path in paths]):
        outcomes[at] = outcome if isinstance(outcome, Refused) else _described(outcome, run.job.limits)
        run.advanced()
    return [(path, outcomes[at]) for at, path in enumerate(paths)]


def _batch_stored(run: _Run, reading: FileReading, batch: Sequence[str]) -> Tally:
    job, files, tally = run.job, run.files, Tally()
    read = _read(run, reading, [path for path in batch if kind_of(path) is not FileKind.NOTE])
    with writing_documents(job.connection, job.limits.batches) as transaction:
        whole = [path for path, outcome in read if _whole(outcome)]
        for path, outcome in read:
            tally += _stored(transaction, path, files[path], outcome)
        for path in (path for path in batch if kind_of(path) is FileKind.NOTE):
            match stored_note(job, transaction, path, files[path], run.identities.get(path)):
                case Tally() as stored:
                    tally += stored
                    whole.append(path)
                case refused:
                    tally += _refused(transaction, path, files[path], refused)
            run.advanced()
        clear_unreadable(transaction, whole)
    for path, outcome in read:
        if isinstance(outcome, FileRead) and outcome.problems:
            job.report(Noted(f"{path}: {PROBLEMS_APART.join(outcome.problems)}"))
    return tally


def _read_and_stored(job: Job, plan: IndexPlan, files: Mapping[str, Stat], identities: Mapping[str, Identity]) -> Tally:
    tally, limits, run = Tally(), job.limits, _Run(job, files, identities)
    for path in (path for path in plan.read if streamed(path)):
        tally += stored_records(job, path, files[path], job.collection.choices.records, identities.get(path))
        run.advanced()
    whole = tuple(path for path in plan.read if not streamed(path))
    with job.readers.files() as reading:
        for batch in batches(whole, files, limits.most_files_a_write, limits.most_bytes_a_write):
            tally += _batch_stored(run, reading, batch)
    return tally


def _whole(outcome: FileRead | Refused) -> bool:
    return isinstance(outcome, FileRead) and not outcome.problems


def _synced(job: Job, ignore: Ignore) -> Indexed:
    job.report(PhaseStarted(Phase.SCAN, None))
    started = job.readers.clock()
    scanned = scan(
        job.collection.folder,
        Reach(job.collection.choices.follow_links, ignore),
        progress=lambda found: job.report(Advanced(found)),
    )
    files = recorded(scanned.found.files, started)
    job.report(PhaseDone(len(files)))
    with reading(job.connection) as transaction:
        known = {path: Stat(source.modified_ns, source.size) for path, source in sources(transaction).items()}
        marks = {entry.path: bool(entry.read_again) for entry in unreadable(transaction)}
    plan = index_plan(scanned.found, known, job.limits.most_file_bytes, marks, ignore)
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


def _ignoring(job: Job) -> CollectionIndexed:
    try:
        ignore = ignore_at(job.collection.folder)
    except OSError as error:
        reason = f"its {IGNORE_FILE} can't be read ({error.strerror or error})"
        return _skipped(job.collection, job.report, reason, IGNORE_REMEDY)
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
            return _ignoring(Job(one, connection, report, limits, readers))
    except StoreUnavailableError as error:
        return _skipped(one, report, error.message, error.remedy)
    except OSError as error:
        return _skipped(one, report, f"it is out of reach ({error.strerror or error})", OUT_OF_REACH_REMEDY)


def index(
    home: Path, request: IndexCollections, report: Reporter, limits: Limits = LIMITS, readers: Readers = READERS
) -> IndexResult:
    chosen = chosen_or_active(home, request.names)
    return IndexResult(tuple(_indexed(one, report, limits, readers) for one in chosen))
