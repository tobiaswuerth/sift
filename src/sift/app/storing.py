from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TypeGuard

from sift.api.events import Advanced, Noted
from sift.app.filing import Job, Limits, Tally, too_large_to_describe
from sift.app.records import stored_records
from sift.app.scans import ScanRun, ScansRead, scans_read
from sift.app.streamed import stored_note
from sift.domain.plan.indexing import IndexPlan, Stat, batches
from sift.domain.text.documents import FileRead, Outcome, Refused, text_held, too_large
from sift.domain.text.kinds import FileKind, kind_of, streamed
from sift.ingest.files import Identity
from sift.ingest.readers import ToRead
from sift.ingest.sandbox import FileReading
from sift.storage.database import WritingDocuments, reading
from sift.storage.documents import (
    Mark,
    add_documents,
    clear_unreadable,
    digests_of,
    kept,
    mark_unreadable,
    put_source,
)
from sift.storage.forgetting import delete_documents, delete_sources, documents_at
from sift.storage.kept_vectors import vectors_held, vectors_kept, vectors_let_go
from sift.storage.keywords import writing_documents

PROBLEMS_APART = "; "


def _described(read: FileRead, limits: Limits) -> FileRead:
    found = [(each, too_large_to_describe(each, limits.most_description_bytes)) for each in read.documents]
    kept = tuple(each for each, problem in found if problem is None)
    large = (f"{each.key}: {problem}" if each.key else problem for each, problem in found if problem is not None)
    return replace(read, documents=kept, problems=(*read.problems, *large))


def _refused(
    transaction: WritingDocuments, path: str, stat: Stat, refused: Refused, release: Callable[[], str]
) -> Tally:
    under = release() if refused.stopped else None
    mark_unreadable(transaction, path, Mark(refused.problem, refused.read_again, stopped_under=under))
    if refused.read_again:
        return Tally()
    source = put_source(transaction, path, stat.modified_ns, stat.size)
    if refused.stopped:
        return Tally()
    gone = [document.id for document in kept(transaction, source)]
    delete_documents(transaction, gone)
    return Tally(removed=len(gone))


def _stored(
    transaction: WritingDocuments, path: str, stat: Stat, read: FileRead | Refused, release: Callable[[], str]
) -> Tally:
    match read:
        case Refused():
            return _refused(transaction, path, stat, read, release)
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
        under = None if read.skipped_under is None else read.skipped_under.value
        problem = PROBLEMS_APART.join(read.problems)
        mark_unreadable(transaction, path, Mark(problem, read.read_again, skipped_under=under))
    return Tally(added=len(new), changed=len(changed), removed=len(gone))


def dropped(job: Job, plan: IndexPlan, files: Mapping[str, Stat]) -> Tally:
    most = job.limits.most_file_bytes
    with writing_documents(job.connection, job.limits.batches) as transaction:
        vectors_let_go(transaction)
        vectors_held(transaction, documents_at(transaction, plan.drop))
        removed = delete_sources(transaction, plan.drop)
        clear_unreadable(transaction, plan.forget)
        release = job.readers.release
        large = [
            _refused(transaction, path, files[path], too_large(files[path].size, most), release)
            for path in plan.too_large
        ]
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


def _scanned(outcome: Outcome) -> TypeGuard[FileRead]:
    return isinstance(outcome, FileRead) and bool(outcome.scans)


def _finished(run: _Run, outcome: Outcome) -> Outcome:
    return outcome if isinstance(outcome, Refused) else _described(outcome, run.job.limits)


def _scans_finished(run: _Run, scans: ScanRun, path: str, read: FileRead) -> ScansRead:
    with reading(run.job.connection) as transaction:
        held = digests_of(transaction, path)
    found = scans_read(scans, path, read, held)
    return replace(found, read=_described(found.read, run.job.limits))


def _read_and_written(run: _Run, read_files: FileReading, paths: Sequence[str]) -> tuple[Tally, list[Outcome]]:
    tally, plain, held = Tally(), list[tuple[str, Outcome]](), 0
    outcomes: dict[int, Outcome] = {}
    for at, outcome in read_files([run.asked(path) for path in paths]):
        outcomes[at] = outcome
        if _scanned(outcome):
            continue
        run.advanced()
        finished = _finished(run, outcome)
        plain.append((paths[at], finished))
        held += text_held(finished)
        if held >= run.job.limits.most_characters_a_write:
            tally += _written(run, plain, ())
            plain, held = [], 0
    tally += _written(run, plain, ())
    return tally, [outcomes[at] for at in range(len(paths))]


def _written(run: _Run, read: Sequence[tuple[str, Outcome]], notes: Sequence[str]) -> Tally:
    job, files, tally = run.job, run.files, Tally()
    if not read and not notes:
        return tally
    release = job.readers.release
    with writing_documents(job.connection, job.limits.batches) as transaction:
        whole = [path for path, outcome in read if _whole(outcome)]
        for path, outcome in read:
            tally += _stored(transaction, path, files[path], outcome, release)
        for path in notes:
            match stored_note(job, transaction, path, files[path], run.identities.get(path)):
                case Tally() as stored:
                    tally += stored
                    whole.append(path)
                case refused:
                    tally += _refused(transaction, path, files[path], refused, release)
            run.advanced()
        clear_unreadable(transaction, whole)
    for path, outcome in read:
        if isinstance(outcome, FileRead) and outcome.problems:
            job.report(Noted(f"{path}: {PROBLEMS_APART.join(outcome.problems)}"))
    return tally


def _batch_stored(run: _Run, read_files: FileReading, scans: ScanRun, batch: Sequence[str]) -> Tally:
    paths = [path for path in batch if kind_of(path) is not FileKind.NOTE]
    tally, read = _read_and_written(run, read_files, paths)
    tally += _written(run, (), [path for path in batch if kind_of(path) is FileKind.NOTE])
    held: list[tuple[str, Outcome]] = []
    for path, outcome in ((path, outcome) for path, outcome in zip(paths, read, strict=True) if _scanned(outcome)):
        found = _scans_finished(run, scans, path, outcome)
        if found.model_ready:
            tally += _written(run, [(path, found.read)], ())
            run.advanced()
        else:
            held.append((path, found.read))
    tally += _written(run, held, ())
    for _ in held:
        run.advanced()
    return tally


def read_and_stored(job: Job, plan: IndexPlan, files: Mapping[str, Stat], identities: Mapping[str, Identity]) -> Tally:
    tally, limits, run = Tally(), job.limits, _Run(job, files, identities)
    for path in (path for path in plan.read if streamed(path)):
        tally += stored_records(job, path, files[path], job.collection.choices.records, identities.get(path))
        run.advanced()
    whole = tuple(path for path in plan.read if not streamed(path))
    with job.readers.files() as read_files, job.readers.scanning.pictures() as picturing:
        scans = ScanRun(job.ocr, picturing, job.report)
        for batch in batches(whole, files, limits.most_files_a_write, limits.most_bytes_a_write):
            tally += _batch_stored(run, read_files, scans, batch)
    return tally


def _whole(outcome: FileRead | Refused) -> bool:
    return isinstance(outcome, FileRead) and not outcome.problems
