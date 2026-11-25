from typing import NamedTuple

from sift.storage.database import Reading, Value, first, rows

PAST_FOLDER = "0"
FIRST_IN_A_FOLDER = """SELECT path FROM sources
                       WHERE path >= :at AND (:past IS NULL OR path < :past) AND instr(substr(path, :after), '/') > 0
                       ORDER BY path LIMIT 1"""
DOCUMENTS = """SELECT s.path, d.key, d.title FROM sources s JOIN documents d ON d.source = s.id
               WHERE s.path >= :at AND (:past IS NULL OR s.path < :past)
               ORDER BY s.path, d.key LIMIT :most"""


class ListedRow(NamedTuple):
    path: str
    key: str
    title: str


def _past(folder: str) -> str | None:
    return folder[:-1] + PAST_FOLDER if folder else None


def folders_in(transaction: Reading, folder: str) -> list[str]:
    found, asked = list[str](), {"at": folder, "past": _past(folder), "after": len(folder) + 1}
    while (row := first(transaction, Value, FIRST_IN_A_FOLDER, asked)) is not None:
        name = row.value[len(folder) :].split("/", 1)[0]
        found.append(name)
        asked["at"] = f"{folder}{name}{PAST_FOLDER}"
    return sorted(found)


def documents_in(transaction: Reading, folder: str, most: int) -> list[ListedRow]:
    return rows(transaction, ListedRow, DOCUMENTS, {"at": folder, "past": _past(folder), "most": most})
