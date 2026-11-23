import os
import stat
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Protocol

from sift.domain.plan.indexing import Found, Stat, stored_time
from sift.domain.text.notes import TEXT_FILES
from sift.domain.text.records import RECORD_FILES

SKIPPED_FOLDERS = frozenset({"node_modules", "__pycache__"})
GONE = (FileNotFoundError, NotADirectoryError)
PROGRESS_EVERY = 1000


class Entry(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def path(self) -> str: ...
    def is_dir(self, *, follow_symlinks: bool = True) -> bool: ...
    def is_junction(self) -> bool: ...
    def is_symlink(self) -> bool: ...
    def stat(self) -> os.stat_result: ...


type Listing = Callable[[str], AbstractContextManager[Iterable[Entry]]]
type Progress = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class Scanned:
    found: Found
    unnamed: tuple[str, ...] = ()
    refusals: Mapping[str, str] = field(default_factory=dict[str, str])


@dataclass(slots=True)
class _Seen:
    files: dict[str, Stat] = field(default_factory=dict[str, Stat])
    unlisted_folders: list[str] = field(default_factory=list[str])
    unlisted_files: list[str] = field(default_factory=list[str])
    outside: list[str] = field(default_factory=list[str])
    unnamed: list[str] = field(default_factory=list[str])
    refusals: dict[str, str] = field(default_factory=dict[str, str])
    empty_folders: list[str] = field(default_factory=list[str])


def _reason(error: OSError) -> str:
    return error.strerror or str(error)


def _storable(relative: str) -> bool:
    try:
        relative.encode()
    except UnicodeEncodeError:
        return False
    return True


def _folder_taken(entry: Entry) -> bool:
    return not entry.name.startswith(".") and entry.name not in SKIPPED_FOLDERS and not entry.is_junction()


def _unreached(seen: _Seen, relative: str, error: OSError) -> None:
    seen.unlisted_files.append(relative)
    seen.refusals[relative] = _reason(error)


def _leads_outside(entry: Entry, outside_of: Path | None) -> bool:
    return outside_of is not None and entry.is_symlink() and not Path(entry.path).resolve().is_relative_to(outside_of)


def _broken_link(entry: Entry, relative: str, seen: _Seen, outside_of: Path | None, error: OSError) -> None:
    if _leads_outside(entry, outside_of):
        seen.outside.append(relative)
    elif entry.is_symlink():
        _unreached(seen, relative, error)


def _file(entry: Entry, relative: str, seen: _Seen, outside_of: Path | None) -> None:
    try:
        found = entry.stat()
    except GONE as error:
        _broken_link(entry, relative, seen, outside_of, error)
        return
    except OSError as error:
        _unreached(seen, relative, error)
        return
    if not stat.S_ISREG(found.st_mode):
        return
    if _leads_outside(entry, outside_of):
        seen.outside.append(relative)
    elif not _storable(relative):
        seen.unnamed.append(relative.encode(errors="backslashreplace").decode())
    else:
        seen.files[relative] = Stat(stored_time(found.st_mtime_ns), found.st_size)


def _listed(folder: str, relative: str, listing: Listing, seen: _Seen) -> Iterator[Entry]:
    try:
        entries = listing(folder)
    except GONE:
        return
    except OSError as error:
        seen.unlisted_folders.append(relative)
        seen.refusals[relative] = _reason(error)
        return
    empty = True
    with entries as listed:
        try:
            for entry in listed:
                empty = False
                yield entry
        except OSError as error:
            seen.unlisted_folders.append(relative)
            seen.refusals[relative] = _reason(error)
            return
    if empty and relative:
        seen.empty_folders.append(relative)


def _taken(entry: Entry, relative: str, seen: _Seen, folders: list[tuple[str, str]], outside_of: Path | None) -> None:
    try:
        if entry.is_dir(follow_symlinks=False):
            if _folder_taken(entry):
                folders.append((entry.path, f"{relative}{entry.name}/"))
        elif PurePath(entry.name).suffix.lower() in (*TEXT_FILES, *RECORD_FILES):
            _file(entry, relative + entry.name, seen, outside_of)
    except OSError as error:
        _unreached(seen, relative + entry.name, error)
        seen.unlisted_folders.append(f"{relative}{entry.name}/")
        seen.refusals[f"{relative}{entry.name}/"] = _reason(error)


def scan(
    root: Path, *, follow_links: bool, listing: Listing = os.scandir, progress: Progress = lambda _: None
) -> Scanned:
    seen, outside_of = _Seen(), None if follow_links else root.resolve()
    folders, looked_at = [(str(root), "")], 0
    while folders:
        folder, relative = folders.pop()
        for entry in _listed(folder, relative, listing, seen):
            _taken(entry, relative, seen, folders, outside_of)
            looked_at += 1
            if looked_at % PROGRESS_EVERY == 0:
                progress(len(seen.files))
    found = Found(
        seen.files,
        tuple(seen.unlisted_folders),
        tuple(seen.unlisted_files),
        tuple(seen.outside),
        tuple(seen.empty_folders),
    )
    return Scanned(found, tuple(seen.unnamed), MappingProxyType(seen.refusals))
