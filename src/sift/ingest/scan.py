import os
import stat
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Protocol

from sift.domain.plan.indexing import Found, Stat, stored_time
from sift.domain.text.decoding import encodable
from sift.domain.text.notes import TEXT_FILES
from sift.domain.text.records import RECORD_FILES, split_record_path
from sift.ingest.files import CHANGED, ChangedError, Identity, identity_at, own_identity_at
from sift.ingest.links import LinkTarget, lands_within, link_target

SKIPPED_FOLDERS = frozenset({"node_modules", "__pycache__"})
GONE = (FileNotFoundError, NotADirectoryError)
PROGRESS_EVERY = 1000
HIDDEN_SYSTEM = stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM


class DirEntry(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def path(self) -> str: ...
    def is_dir(self, *, follow_symlinks: bool = True) -> bool: ...
    def is_junction(self) -> bool: ...
    def is_symlink(self) -> bool: ...
    def stat(self) -> os.stat_result: ...


type Listing = Callable[[str], AbstractContextManager[Iterable[DirEntry]]]
type Progress = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class LinkChecks:
    target: LinkTarget = link_target
    identity: Callable[[Path], Identity] = identity_at
    own_identity: Callable[[Path], Identity] = own_identity_at


LINK_CHECKS = LinkChecks()


@dataclass(frozen=True, slots=True)
class _Guard:
    root: Path
    checks: LinkChecks


@dataclass(frozen=True, slots=True)
class Scanned:
    found: Found
    unnamed: tuple[str, ...] = ()
    refusals: Mapping[str, str] = field(default_factory=dict[str, str])
    identities: Mapping[str, Identity] = field(default_factory=dict[str, Identity])
    linked_folders: tuple[str, ...] = ()
    named_as_records: tuple[str, ...] = ()


@dataclass(slots=True)
class _Seen:
    files: dict[str, Stat] = field(default_factory=dict[str, Stat])
    unlisted_folders: list[str] = field(default_factory=list[str])
    unlisted_files: list[str] = field(default_factory=list[str])
    outside: list[str] = field(default_factory=list[str])
    unnamed: list[str] = field(default_factory=list[str])
    refusals: dict[str, str] = field(default_factory=dict[str, str])
    empty_folders: list[str] = field(default_factory=list[str])
    linked_folders: list[str] = field(default_factory=list[str])
    named_as_records: list[str] = field(default_factory=list[str])
    identities: dict[str, Identity] = field(default_factory=dict[str, Identity])


def _reason(error: OSError) -> str:
    return error.strerror or str(error)


def _system(entry: DirEntry) -> bool:
    return sys.platform == "win32" and entry.stat().st_file_attributes & HIDDEN_SYSTEM == HIDDEN_SYSTEM


def _folder_taken(entry: DirEntry) -> bool:
    return not entry.name.startswith(".") and entry.name not in SKIPPED_FOLDERS and not _system(entry)


def _unreached(seen: _Seen, relative: str, error: OSError) -> None:
    seen.unlisted_files.append(relative)
    seen.refusals[relative] = _reason(error)


def _guarded(entry: DirEntry, relative: str, guard: _Guard) -> Identity | None:
    path = Path(entry.path)
    if not entry.is_symlink():
        return guard.checks.own_identity(path)
    if not lands_within(relative, guard.root, guard.checks.target):
        return None
    identity = guard.checks.identity(path)
    if not lands_within(relative, guard.root, guard.checks.target):
        raise ChangedError(CHANGED)
    return identity


def _file(entry: DirEntry, relative: str, seen: _Seen, guard: _Guard | None) -> None:
    try:
        identity = None if guard is None else _guarded(entry, relative, guard)
        if guard is not None and identity is None:
            seen.outside.append(relative)
            return
        found = entry.stat()
    except GONE as error:
        if entry.is_symlink():
            _unreached(seen, relative, error)
        return
    except OSError as error:
        _unreached(seen, relative, error)
        return
    if not stat.S_ISREG(found.st_mode):
        return
    if not encodable(relative):
        seen.unnamed.append(relative.encode(errors="backslashreplace").decode())
        return
    if split_record_path(relative)[1]:
        seen.named_as_records.append(relative)
        return
    seen.files[relative] = Stat(stored_time(found.st_mtime_ns), found.st_size)
    if identity is not None:
        seen.identities[relative] = identity


def _listed(folder: str, relative: str, listing: Listing, seen: _Seen) -> Iterator[DirEntry]:
    try:
        entries = listing(folder)
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


def _taken(entry: DirEntry, relative: str, seen: _Seen, folders: list[tuple[str, str]], guard: _Guard | None) -> None:
    try:
        if entry.is_dir(follow_symlinks=False):
            if entry.is_junction():
                seen.linked_folders.append(f"{relative}{entry.name}/")
            elif _folder_taken(entry):
                folders.append((entry.path, f"{relative}{entry.name}/"))
        elif PurePath(entry.name).suffix.lower() in (*TEXT_FILES, *RECORD_FILES):
            _file(entry, relative + entry.name, seen, guard)
    except OSError as error:
        _unreached(seen, relative + entry.name, error)
        seen.unlisted_folders.append(f"{relative}{entry.name}/")
        seen.refusals[f"{relative}{entry.name}/"] = _reason(error)


def scan(
    root: Path,
    *,
    follow_links: bool,
    listing: Listing = os.scandir,
    links: LinkChecks = LINK_CHECKS,
    progress: Progress = lambda _: None,
) -> Scanned:
    seen, guard = _Seen(), None if follow_links else _Guard(root.resolve(), links)
    folders, looked_at = [(str(root), "")], 0
    while folders:
        folder, relative = folders.pop()
        for entry in _listed(folder, relative, listing, seen):
            _taken(entry, relative, seen, folders, guard)
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
    return Scanned(
        found,
        tuple(seen.unnamed),
        MappingProxyType(seen.refusals),
        MappingProxyType(seen.identities),
        tuple(seen.linked_folders),
        tuple(seen.named_as_records),
    )
