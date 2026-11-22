from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

MOST_TIME = 2**63 - 1
MOST_TEXT_BYTES = 1_000_000_000
ROOT = ""


class Stat(NamedTuple):
    modified_ns: int
    size: int


def stored_time(modified_ns: int) -> int:
    return max(-MOST_TIME, min(MOST_TIME, modified_ns))


@dataclass(frozen=True, slots=True)
class Found:
    files: Mapping[str, Stat]
    unlisted_folders: tuple[str, ...] = ()
    unlisted_files: tuple[str, ...] = ()
    outside: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IndexPlan:
    read: tuple[str, ...]
    too_large: tuple[str, ...]
    drop: tuple[str, ...]
    kept: tuple[str, ...]
    left_as_indexed: tuple[str, ...]
    unchanged: int


def _in_folder(path: str, folder: str) -> bool:
    return folder == ROOT or path.startswith(folder)


def _kept(unseen: frozenset[str], found: Found) -> frozenset[str]:
    in_folders = {path for path in unseen for folder in found.unlisted_folders if _in_folder(path, folder)}
    return frozenset(in_folders | (unseen & set(found.unlisted_files)))


def index_plan(found: Found, known: Mapping[str, Stat], most_text_bytes: int = MOST_TEXT_BYTES) -> IndexPlan:
    new_or_changed = sorted(path for path, stat in found.files.items() if known.get(path) != stat)
    unseen = frozenset(known.keys() - found.files.keys())
    kept = _kept(unseen, found)
    gone = unseen - kept
    nothing_there = not found.files and not found.outside
    left_as_indexed = gone if nothing_there else frozenset[str]()
    return IndexPlan(
        read=tuple(path for path in new_or_changed if found.files[path].size <= most_text_bytes),
        too_large=tuple(path for path in new_or_changed if found.files[path].size > most_text_bytes),
        drop=tuple(sorted(gone - left_as_indexed)),
        kept=tuple(sorted(kept)),
        left_as_indexed=tuple(sorted(left_as_indexed)),
        unchanged=len(found.files) - len(new_or_changed),
    )


def batches(
    paths: Sequence[str], stats: Mapping[str, Stat], most_files: int, most_bytes: int
) -> tuple[tuple[str, ...], ...]:
    done: list[tuple[str, ...]] = []
    batch: list[str] = []
    held = 0
    for path in paths:
        size = stats[path].size
        if batch and (len(batch) == most_files or held + size > most_bytes):
            done.append(tuple(batch))
            batch, held = [], 0
        batch.append(path)
        held += size
    return (*done, tuple(batch)) if batch else tuple(done)
