from typing import NamedTuple

from sift.api.ids import DocumentId
from sift.storage.database import Reading, Value, Writing, first, rows, run
from sift.storage.documents import SourceId

RETITLING = "retitling"
SPAN = "SELECT count(*), coalesce(max(id), 0) FROM documents WHERE key = ''"
NOTES = """SELECT d.id, d.source, s.path, d.title, d.digest FROM documents d JOIN sources s ON s.id = d.source
           WHERE d.key = '' AND d.id > ? AND d.id <= ? ORDER BY d.id LIMIT ?"""


class TitledNote(NamedTuple):
    id: DocumentId
    source: SourceId
    path: str
    title: str
    digest: str


class WholeFiles(NamedTuple):
    files: int
    last: DocumentId


def whole_files(transaction: Reading) -> WholeFiles:
    (found,) = rows(transaction, WholeFiles, SPAN)
    return found


def titled_after(transaction: Reading, after: DocumentId, last: DocumentId, most: int) -> list[TitledNote]:
    return rows(transaction, TitledNote, NOTES, (after, last, most))


def promise(transaction: Writing, choices: str) -> None:
    run(transaction, "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (RETITLING, choices))


def promised(transaction: Reading) -> str | None:
    found = first(transaction, Value, "SELECT value FROM meta WHERE key = ?", (RETITLING,))
    return None if found is None else str(found.value)


def fulfilled(transaction: Writing) -> None:
    run(transaction, "DELETE FROM meta WHERE key = ?", (RETITLING,))
