from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

from sift.domain.text.kinds import CONVERTED, kind_of, streamed

MOST_TIME = 2**63 - 1
RACY = -1
GRAIN_NS = 2 * 10**9
MOST_TEXT_BYTES = 64 << 20
MOST_FILE_BYTES = 1 << 30
ROOT = ""
NONE_UNREADABLE: Mapping[str, bool] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Sizes:
    text: int = MOST_TEXT_BYTES
    file: int = MOST_FILE_BYTES


SIZES = Sizes()


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


def recorded(files: Mapping[str, Stat], scanned_ns: int) -> dict[str, Stat]:
    return {
        path: Stat(RACY, stat.size) if stat.modified_ns >= scanned_ns - GRAIN_NS else stat
        for path, stat in files.items()
    }


def folders_of(path: str) -> Iterator[str]:
    yield ROOT
    at = path.find("/")
    while at >= 0:
        yield path[: at + 1]
        at = path.find("/", at + 1)


def within(paths: frozenset[str], folders: tuple[str, ...]) -> frozenset[str]:
    held = frozenset(folders)
    return frozenset(path for path in paths if any(folder in held for folder in folders_of(path)))


def _emptied(found: Found, gone: frozenset[str]) -> tuple[str, ...]:
    if not found.files and not found.outside:
        return (ROOT,) if gone else ()
    touched = {folder for path in gone for folder in folders_of(path)}
    return tuple(sorted(folder for folder in found.empty_folders if folder in touched))


def most_read(path: str, sizes: Sizes) -> int:
    return sizes.file if kind_of(path) in CONVERTED else sizes.text


def _too_large(path: str, stat: Stat, sizes: Sizes) -> bool:
    return not streamed(path) and stat.size > most_read(path, sizes)


def index_plan(
    found: Found,
    known: Mapping[str, Stat],
    sizes: Sizes = SIZES,
    unreadable: Mapping[str, bool] = NONE_UNREADABLE,
) -> IndexPlan:
    again = {path for path, read_again in unreadable.items() if read_again}
    to_read = sorted(path for path, stat in found.files.items() if known.get(path) != stat or path in again)
    unseen = frozenset(known.keys() - found.files.keys())
    kept = within(unseen, found.unlisted_folders) | (unseen & set(found.unlisted_files))
    gone = unseen - kept
    emptied = _emptied(found, gone)
    left_as_indexed = within(gone, emptied)
    still_there = found.files.keys() | kept | left_as_indexed | set(found.unlisted_files)
    return IndexPlan(
        read=tuple(path for path in to_read if not _too_large(path, found.files[path], sizes)),
        too_large=tuple(path for path in to_read if _too_large(path, found.files[path], sizes)),
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
