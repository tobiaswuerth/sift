from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import batched

from sift.api.events import Noted
from sift.app.filing import Job, Tally
from sift.app.kept_vectors import vectors_held, vectors_kept
from sift.domain.plan.indexing import RACY, Stat
from sift.domain.text.notes import Document
from sift.domain.text.records import Chosen, SkippedLine, record_of
from sift.domain.text.wording import counted
from sift.ingest.files import Identity
from sift.storage.database import WritingDocuments
from sift.storage.documents import (
    SourceId,
    add_documents,
    clear_unreadable,
    mark_unreadable,
    put_source,
)
from sift.storage.forgetting import delete_documents
from sift.storage.keywords import writing_documents
from sift.storage.taken import keyed, nothing_taken, take, untaken

SHOWN = 5

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


def _records(lines: Iterable[tuple[int, str | SkippedLine]], chosen: Chosen, most: int) -> Iterator[Line]:
    for number, line in lines:
        if not line:
            continue
        found = line if isinstance(line, SkippedLine) else record_of(line, number, chosen)
        if isinstance(found, Document) and (size := len(found.text.encode())) > most:
            yield number, SkippedLine(number, f"{size:,} bytes as text, more than the {most:,} a document holds")
        else:
            yield number, found


def _firsts(read: Sequence[Line], new: frozenset[str]) -> tuple[list[Document], list[SkippedLine]]:
    kept: list[Document] = []
    skipped: list[SkippedLine] = []
    used: set[str] = set()
    for number, each in read:
        match each:
            case SkippedLine():
                skipped.append(each)
            case Document() if each.key in new and each.key not in used:
                used.add(each.key)
                kept.append(each)
            case _:
                skipped.append(SkippedLine(number, f"id {each.key} used twice"))
    return kept, skipped


def _batch(
    transaction: WritingDocuments, source: SourceId, path: str, read: Sequence[Line]
) -> tuple[Tally, list[SkippedLine]]:
    keys = [each.key for _, each in read if isinstance(each, Document)]
    documents, skipped = _firsts(read, take(transaction, keys))
    held = keyed(transaction, source, [each.key for each in documents])
    changed = [each for each in documents if each.key in held and held[each.key].digest != each.digest]
    new = [each for each in documents if each.key not in held]
    gone = [held[each.key].id for each in changed]
    vectors = vectors_held(transaction, gone)
    delete_documents(transaction, gone)
    added = add_documents(transaction, source, path, [*changed, *new])
    vectors_kept(transaction, added[: len(changed)], vectors)
    return Tally(added=len(new), changed=len(changed)), skipped


def _finished(job: Job, path: str, stat: Stat, source: SourceId, skips: Skips) -> int:
    problem = skips.said()
    with writing_documents(job.connection, job.limits.batches) as transaction:
        gone = untaken(transaction, source)
        delete_documents(transaction, gone)
        put_source(transaction, path, stat.modified_ns, stat.size)
        if problem is None:
            clear_unreadable(transaction, [path])
        else:
            mark_unreadable(transaction, path, problem, read_again=False)
    if problem is not None:
        job.report(Noted(f"{path}: {problem}"))
    return len(gone)


def stored_records(job: Job, path: str, stat: Stat, chosen: Chosen, scanned: Identity | None) -> Tally:
    limits, tally, skips = job.limits, Tally(), Skips()
    lines = job.readers.lines(job.collection.folder / path, limits.most_text_bytes, scanned)
    with writing_documents(job.connection, limits.batches) as transaction:
        source = put_source(transaction, path, RACY, stat.size)
        nothing_taken(transaction)
    try:
        for read in batched(_records(lines, chosen, limits.most_text_bytes), limits.most_records_a_write):
            with writing_documents(job.connection, limits.batches) as transaction:
                done, skipped = _batch(transaction, source, path, read)
            tally, skips = tally + done, skips + skipped
    except OSError as error:
        with writing_documents(job.connection, limits.batches) as transaction:
            mark_unreadable(transaction, path, error.strerror or str(error), read_again=True)
        return tally
    return tally + Tally(removed=_finished(job, path, stat, source, skips))
