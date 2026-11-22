from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from sift.api.errors import StoreUnavailableError
from sift.storage.database import Reading, Writing, rows, run

type Migration = tuple[str, ...]


class Version(NamedTuple):
    number: int


def version(transaction: Reading) -> int:
    (found,) = rows(transaction, Version, "PRAGMA user_version")
    return found.number


def newer(path: Path, found: int, known: int) -> StoreUnavailableError:
    return StoreUnavailableError(
        f"{path} was written by a newer sift (layout {found}; this sift knows up to {known})", "upgrade sift"
    )


def migrated(transaction: Writing, migrations: Sequence[Migration], path: Path) -> int:
    start = version(transaction)
    if start > len(migrations):
        raise newer(path, start, len(migrations))
    for migration in migrations[start:]:
        for statement in migration:
            run(transaction, statement)
    if start < len(migrations):
        run(transaction, f"PRAGMA user_version = {len(migrations)}")
    return start
