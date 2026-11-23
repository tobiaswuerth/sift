from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import batched

from sift.api.events import Noted
from sift.app.filing import Job, Tally
from sift.domain.plan.indexing import Stat
from sift.domain.text.notes import Document
from sift.domain.text.records import Chosen, Skipped, record_of
from sift.storage.database import WritingDocuments
from sift.storage.documents import (
    SourceId,
    add_documents,
    clear_unreadable,
    delete_documents,
    mark_unreadable,
    put_source,
)
from sift.storage.keywords import writing_documents
from sift.storage.taken import keyed, nothing_taken, take, untaken

SHOWN = 5

type Line = tuple[int, Document | Skipped]


@dataclass(frozen=True, slots=True)
class Skips:
    count: int = 0
    first: tuple[Skipped, ...] = ()

    def __add__(self, more: Sequence[Skipped]) -> "Skips":
        shown = tuple(sorted((*self.first, *more), key=lambda each: each.line)[:SHOWN])
        return Skips(self.count + len(more), shown)

    def said(self) -> str | None:
        if not self.count:
            return None
        shown = "; ".join(f"line {each.line}: {each.why}" for each in self.first)
        more = "; ..." if self.count > len(self.first) else ""
        return f"{self.count:,} line{'s' if self.count != 1 else ''} skipped ({shown}{more})"


def _records(lines: Iterable[tuple[int, str]], chosen: Chosen, most: int) -> Iterator[Line]:
    for number, line in lines:
        if not line:
            continue
        found = record_of(line, number, chosen)
        if isinstance(found, Document) and (size := len(found.text.encode())) > most:
            yield number, Skipped(number, f"{size:,} bytes as text, more than the {most:,} a document holds")
        else:
            yield number, found


def _firsts(read: Sequence[Line], new: frozenset[str]) -> tuple[list[Document], list[Skipped]]:
    kept: list[Document] = []
    skipped: list[Skipped] = []
    used: set[str] = set()
    for number, each in read:
        if isinstance(each, Skipped):
            skipped.append(each)
        elif each.key in new and each.key not in used:
            used.add(each.key)
            kept.append(each)
        else:
            skipped.append(Skipped(number, f"id {each.key} used twice"))
    return kept, skipped


def _batch(
    transaction: WritingDocuments, source: SourceId, path: str, read: Sequence[Line]
) -> tuple[Tally, list[Skipped]]:
    keys = [each.key for _, each in read if isinstance(each, Document)]
    documents, skipped = _firsts(read, take(transaction, keys))
    held = keyed(transaction, source, [each.key for each in documents])
    changed = [each for each in documents if each.key in held and held[each.key].digest != each.digest]
    new = [each for each in documents if each.key not in held]
    delete_documents(transaction, [held[each.key].id for each in changed])
    add_documents(transaction, source, path, [*changed, *new])
    return Tally(added=len(new), changed=len(changed)), skipped


def stored_records(job: Job, path: str, stat: Stat, chosen: Chosen) -> Tally:
    limits, tally, skips = job.limits, Tally(), Skips()
    lines = job.lines(job.collection.folder / path)
    with writing_documents(job.connection, limits.batches) as transaction:
        source = put_source(transaction, path, stat.modified_ns, stat.size)
        nothing_taken(transaction)
        for read in batched(_records(lines, chosen, limits.most_text_bytes), limits.most_records_a_write):
            done, skipped = _batch(transaction, source, path, read)
            tally, skips = tally + done, skips + skipped
        gone = untaken(transaction, source)
        delete_documents(transaction, gone)
        problem = skips.said()
        if problem is None:
            clear_unreadable(transaction, [path])
        else:
            mark_unreadable(transaction, path, problem, read_again=False)
    if problem is not None:
        job.report(Noted(f"{path}: {problem}"))
    return tally + Tally(removed=len(gone))
