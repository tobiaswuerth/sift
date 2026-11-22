from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

MOST_TIME = 2**63 - 1
MOST_TEXT_BYTES = 1_000_000_000
ROOT = ""
NONE_UNREADABLE: Mapping[str, bool] = MappingProxyType({})


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
    empty_folders: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IndexPlan:
    read: tuple[str, ...]
    too_large: tuple[str, ...]
    drop: tuple[str, ...]
    kept: tuple[str, ...]
    left_as_indexed: tuple[str, ...]
    emptied: tuple[str, ...]
    forget: tuple[str, ...]
    unchanged: int


def _in_folder(path: str, folder: str) -> bool:
    return folder == ROOT or path.startswith(folder)


def _in_any(paths: frozenset[str], folders: tuple[str, ...]) -> frozenset[str]:
    return frozenset(path for path in paths for folder in folders if _in_folder(path, folder))


def _emptied(found: Found, gone: frozenset[str]) -> tuple[str, ...]:
    if not found.files and not found.outside:
        return (ROOT,) if gone else ()
    return tuple(sorted(folder for folder in found.empty_folders if _in_any(gone, (folder,))))


def index_plan(
    found: Found,
    known: Mapping[str, Stat],
    most_text_bytes: int = MOST_TEXT_BYTES,
    unreadable: Mapping[str, bool] = NONE_UNREADABLE,
) -> IndexPlan:
    again = {path for path, read_again in unreadable.items() if read_again}
    to_read = sorted(path for path, stat in found.files.items() if known.get(path) != stat or path in again)
    unseen = frozenset(known.keys() - found.files.keys())
    kept = _in_any(unseen, found.unlisted_folders) | (unseen & set(found.unlisted_files))
    gone = unseen - kept
    emptied = _emptied(found, gone)
    left_as_indexed = _in_any(gone, emptied)
    still_there = found.files.keys() | kept | left_as_indexed | set(found.unlisted_files)
    return IndexPlan(
        read=tuple(path for path in to_read if found.files[path].size <= most_text_bytes),
        too_large=tuple(path for path in to_read if found.files[path].size > most_text_bytes),
        drop=tuple(sorted(gone - left_as_indexed)),
        kept=tuple(sorted(kept)),
        left_as_indexed=tuple(sorted(left_as_indexed)),
        emptied=emptied,
        forget=tuple(sorted(unreadable.keys() - still_there)),
        unchanged=len(found.files) - len(to_read),
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
