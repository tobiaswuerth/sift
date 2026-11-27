from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from sift.api.events import Noted
from sift.app.filing import Job, StoredHere, Tally, too_large_to_describe
from sift.domain.plan.indexing import RACY, Stat
from sift.domain.text.notes import Document
from sift.domain.text.records import Chosen, SkippedLine, record_of, used_twice
from sift.domain.text.wording import counted
from sift.ingest.files import Identity
from sift.storage.database import WritingDocuments
from sift.storage.documents import (
    Mark,
    SourceId,
    clear_unreadable,
    mark_unreadable,
    put_source,
)
from sift.storage.forgetting import delete_documents
from sift.storage.keywords import writing_documents
from sift.storage.replacing import documents_replaced
from sift.storage.taken import keyed, nothing_taken, take, untaken

SHOWN = 5
DESCRIBED_WITHIN = 16

type Line = tuple[int, Document | SkippedLine]


@dataclass(frozen=True, slots=True)
class Skips:
    count: int = 0
    first: tuple[SkippedLine, ...] = ()

    def __add__(self, more: Sequence[SkippedLine]) -> "Skips":
        shown = tuple(sorted((*self.first, *more), key=lambda each: each.line)[:SHOWN])
        return Skips(self.count + len(more), shown)

    def said(self) -> str | None:
        if not self.count:
            return None
        shown = "; ".join(f"line {each.line}: {each.why}" for each in self.first)
        more = "; ..." if self.count > len(self.first) else ""
        return f"{counted(self.count, 'line')} skipped ({shown}{more})"


def _described(found: Document | SkippedLine, number: int, most: int) -> Document | SkippedLine:
    problem = too_large_to_describe(found, most) if isinstance(found, Document) else None
    return found if problem is None else SkippedLine(number, problem)


def _records(lines: Iterable[tuple[int, str | SkippedLine]], chosen: Chosen, most: int) -> Iterator[Line]:
    for number, line in lines:
        if isinstance(line, SkippedLine):
            yield number, line
        elif line:
            found = record_of(line, number, chosen)
            yield number, found if DESCRIBED_WITHIN * len(line) <= most else _described(found, number, most)


def _characters(line: Line) -> int:
    _, found = line
    return len(found.text) if isinstance(found, Document) else 0


def _writes(read: Iterable[Line], most: int, most_characters: int) -> Iterator[list[Line]]:
    batch: list[Line] = []
    held = 0
    for line in read:
        batch.append(line)
        held += _characters(line)
        if len(batch) >= most or held >= most_characters:
            yield batch
            batch, held = [], 0
    if batch:
        yield batch


def _firsts(read: Sequence[Line], new: frozenset[str], chosen: Chosen) -> tuple[list[Document], list[SkippedLine]]:
    kept: list[Document] = []
    skipped: list[SkippedLine] = []
    used: dict[str, Document] = {}
    for number, each in read:
        match each:
            case SkippedLine():
                skipped.append(each)
            case Document() if each.key in new and each.key not in used:
                used[each.key] = each
                kept.append(each)
            case _:
                skipped.append(SkippedLine(number, used_twice(each, used.get(each.key), chosen)))
    return kept, skipped


def _batch(
    transaction: WritingDocuments, source: SourceId, path: str, read: Sequence[Line], chosen: Chosen
) -> tuple[Tally, list[SkippedLine]]:
    keys = [each.key for _, each in read if isinstance(each, Document)]
    documents, skipped = _firsts(read, take(transaction, keys), chosen)
    held = keyed(transaction, source, [each.key for each in documents])
    found = documents_replaced(transaction, source, path, held, documents)
    return Tally(added=len(found.new), changed=len(found.changed)), skipped


def _finished(job: Job, path: str, stat: Stat, source: SourceId, skips: Skips) -> int:
    problem = skips.said()
    with writing_documents(job.connection, job.limits.batches) as transaction:
        gone = untaken(transaction, source)
        delete_documents(transaction, gone)
        put_source(transaction, path, stat.modified_ns, stat.size)
        if problem is None:
            clear_unreadable(transaction, [path])
        else:
            mark_unreadable(transaction, path, Mark(problem, read_again=False))
    if problem is not None:
        job.report(Noted(f"{path}: {problem}"))
    return len(gone)


def stored_records(job: Job, path: str, stat: Stat, chosen: Chosen, scanned: Identity | None) -> Tally:
    limits, tally, skips = job.limits, Tally(), Skips()
    lines = job.readers.lines(job.collection.folder / path, limits.most_record_characters, scanned)
    with writing_documents(job.connection, limits.batches) as transaction:
        source = put_source(transaction, path, RACY, stat.size)
        nothing_taken(transaction)
    try:
        records = _records(lines, chosen, limits.most_description_bytes)
        for read in _writes(records, limits.most_records_a_write, limits.most_characters_a_write):
            with writing_documents(job.connection, limits.batches) as transaction:
                done, skipped = _batch(transaction, source, path, read, chosen)
            tally, skips = tally + done, skips + skipped
    except OSError as error:
        with writing_documents(job.connection, limits.batches) as transaction:
            mark_unreadable(transaction, path, Mark(error.strerror or str(error), read_again=True))
        return tally
    return tally + Tally(removed=_finished(job, path, stat, source, skips))


def records_stored(
    job: Job, transaction: WritingDocuments, path: str, stat: Stat, scanned: Identity | None
) -> StoredHere:
    limits, chosen, tally, skips = job.limits, job.collection.choices.records, Tally(), Skips()
    source = put_source(transaction, path, stat.modified_ns, stat.size)
    nothing_taken(transaction)
    try:
        lines = job.readers.lines(job.collection.folder / path, limits.most_record_characters, scanned)
        records = _records(lines, chosen, limits.most_description_bytes)
        for read in _writes(records, limits.most_records_a_write, limits.most_characters_a_write):
            done, skipped = _batch(transaction, source, path, read, chosen)
            tally, skips = tally + done, skips + skipped
    except OSError as error:
        return StoredHere(tally, error.strerror or str(error), read_again=True)
    gone = untaken(transaction, source)
    delete_documents(transaction, gone)
    return StoredHere(tally + Tally(removed=len(gone)), skips.said())
