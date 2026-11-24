from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

from sift.api.errors import Problem, StoreUnavailableError
from sift.storage.database import Reading, Writing, rows, run

type Step = str | Callable[[Writing], None]
type Migration = tuple[Step, ...]


class Version(NamedTuple):
    number: int


def version(transaction: Reading) -> int:
    (found,) = rows(transaction, Version, "PRAGMA user_version")
    return found.number


def newer(path: Path, found: int, known: int) -> StoreUnavailableError:
    return StoreUnavailableError(
        f"{path} was written by a newer sift (layout {found}; this sift knows up to {known})",
        "upgrade sift",
        code=Problem.STORE_NEWER,
    )


def _taken(transaction: Writing, step: Step) -> None:
    if isinstance(step, str):
        run(transaction, step)
    else:
        step(transaction)


def migrated(transaction: Writing, migrations: Sequence[Migration], path: Path) -> int:
    start = version(transaction)
    if start > len(migrations):
        raise newer(path, start, len(migrations))
    for migration in migrations[start:]:
        for step in migration:
            _taken(transaction, step)
    if start < len(migrations):
        run(transaction, f"PRAGMA user_version = {len(migrations)}")
    return start
