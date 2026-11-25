import os
import stat
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol

from sift.domain.plan.ignore import NOTHING_LEFT_OUT, Ignore, left_out
from sift.domain.plan.indexing import Found, Stat, stored_time
from sift.domain.plan.left_out import WhyLeftOut
from sift.domain.text.decoding import encodable
from sift.domain.text.kinds import kind_of, split_keyed_path, windows_cannot_open
from sift.ingest.files import CHANGED, ChangedError, Identity, identity_at, own_identity_at
from sift.ingest.links import Landing, LinkTarget, landing, lands_within, link_target

SKIPPED_FOLDERS = frozenset({"node_modules", "__pycache__"})
GONE = (FileNotFoundError, NotADirectoryError)
PROGRESS_EVERY = 1000
HIDDEN_SYSTEM = stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM
UNOPENABLE = "Windows cannot open a folder whose name ends in a dot or a space: rename it"


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
class Reach:
    follow_links: bool = False
    ignore: Ignore = NOTHING_LEFT_OUT
    platform: str = sys.platform


WITHIN = Reach()


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
    looped: tuple[str, ...] = ()
    why: Mapping[str, WhyLeftOut] = field(default_factory=dict[str, WhyLeftOut])


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
    left_out: int = 0
    looped: list[str] = field(default_factory=list[str])
    why: dict[str, WhyLeftOut] | None = None


def _because(seen: _Seen, path: str, reason: WhyLeftOut) -> None:
    if seen.why is not None:
        seen.why.setdefault(path, reason)


@dataclass(frozen=True, slots=True)
class _Walk:
    seen: _Seen
    folders: list[tuple[str, str]]
    guard: _Guard | None
    reach: Reach


def _reason(error: OSError) -> str:
    return error.strerror or str(error)


def _system(entry: DirEntry) -> bool:
    return sys.platform == "win32" and entry.stat().st_file_attributes & HIDDEN_SYSTEM == HIDDEN_SYSTEM


def _folder_taken(entry: DirEntry) -> bool:
    return not entry.name.startswith(".") and entry.name not in SKIPPED_FOLDERS and not _system(entry)


def _unreached(seen: _Seen, relative: str, error: OSError) -> None:
    seen.unlisted_files.append(relative)
    seen.refusals[relative] = _reason(error)


def _guarded(entry: DirEntry, relative: str, guard: _Guard) -> Identity:
    path = Path(entry.path)
    if not entry.is_symlink():
        return guard.checks.own_identity(path)
    identity = guard.checks.identity(path)
    if not lands_within(relative, guard.root, guard.checks.target):
        raise ChangedError(CHANGED)
    return identity


def _file(entry: DirEntry, relative: str, seen: _Seen, guard: _Guard | None) -> None:
    try:
        identity = None if guard is None else _guarded(entry, relative, guard)
        found = entry.stat()
    except GONE as error:
        if entry.is_symlink():
            _unreached(seen, relative, error)
        return
    except OSError as error:
        _unreached(seen, relative, error)
        return
    if not stat.S_ISREG(found.st_mode):
        _because(seen, relative, WhyLeftOut.NO_FILE)
        return
    if not encodable(relative):
        seen.unnamed.append(relative.encode(errors="backslashreplace").decode())
        _because(seen, relative, WhyLeftOut.UNNAMED)
        return
    if split_keyed_path(relative)[1]:
        seen.named_as_records.append(relative)
        _because(seen, relative, WhyLeftOut.RECORD_NAME)
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


def _left_out(walk: _Walk, path: str, *, folder: bool = False) -> bool:
    found = left_out(walk.reach.ignore, path, folder=folder)
    walk.seen.left_out += found
    if found:
        _because(walk.seen, path, WhyLeftOut.RULES)
    return found


def _links_to_a_folder(entry: DirEntry) -> bool:
    try:
        return entry.is_symlink() and entry.is_dir()
    except OSError:
        return False


def _linked_folder(path: str, walk: _Walk) -> None:
    if not _left_out(walk, path, folder=True):
        walk.seen.linked_folders.append(f"{path}/")
        _because(walk.seen, path, WhyLeftOut.LINKED_FOLDER)


def _unopenable(path: str, walk: _Walk) -> None:
    if not _left_out(walk, path, folder=True):
        walk.seen.unlisted_folders.append(f"{path}/")
        walk.seen.refusals[f"{path}/"] = UNOPENABLE


def _folder(entry: DirEntry, path: str, walk: _Walk) -> None:
    if entry.is_junction():
        _linked_folder(path, walk)
    elif windows_cannot_open(entry.name, walk.reach.platform):
        _unopenable(path, walk)
    elif _left_out(walk, path, folder=True):
        pass
    elif _folder_taken(entry):
        walk.folders.append((entry.path, f"{path}/"))
    else:
        _because(walk.seen, path, WhyLeftOut.PASSED_OVER)


def _untyped(path: str, walk: _Walk, error: OSError) -> None:
    if not _left_out(walk, path):
        _unreached(walk.seen, path, error)
    if not _left_out(walk, path, folder=True):
        walk.seen.unlisted_folders.append(f"{path}/")
        walk.seen.refusals[f"{path}/"] = _reason(error)


def _led_away(path: str, walk: _Walk, landed: Landing) -> None:
    walk.seen.outside.append(path)
    if landed is Landing.LOOP:
        walk.seen.looped.append(path)
    _because(walk.seen, path, WhyLeftOut.LOOP if landed is Landing.LOOP else WhyLeftOut.OUTSIDE)


def _link(entry: DirEntry, path: str, walk: _Walk) -> None:
    guard, readable = walk.guard, kind_of(entry.name) is not None
    landed = Landing.WITHIN if guard is None else landing(path, guard.root, guard.checks.target)
    if landed is not Landing.WITHIN:
        if readable and not _left_out(walk, path):
            _led_away(path, walk, landed)
    elif _links_to_a_folder(entry):
        _linked_folder(path, walk)
    elif not readable:
        _because(walk.seen, path, WhyLeftOut.NO_SUCH_KIND)
    elif not _left_out(walk, path):
        _file(entry, path, walk.seen, guard)


def _taken(entry: DirEntry, relative: str, walk: _Walk) -> None:
    path = relative + entry.name
    try:
        if entry.is_dir(follow_symlinks=False):
            _folder(entry, path, walk)
        elif entry.is_symlink():
            _link(entry, path, walk)
        elif kind_of(entry.name) is None:
            _because(walk.seen, path, WhyLeftOut.NO_SUCH_KIND)
        elif not _left_out(walk, path):
            _file(entry, path, walk.seen, walk.guard)
    except OSError as error:
        _untyped(path, walk, error)


@dataclass(frozen=True, slots=True)
class _PathEntry:
    location: Path

    @property
    def name(self) -> str:
        return self.location.name

    @property
    def path(self) -> str:
        return str(self.location)

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        return self.location.is_dir() and (follow_symlinks or not self.location.is_symlink())

    def is_junction(self) -> bool:
        return self.location.is_junction()

    def is_symlink(self) -> bool:
        return self.location.is_symlink()

    def stat(self) -> os.stat_result:
        return self.location.stat()


def _passed_over(root: Path, folder: str, walk: _Walk) -> WhyLeftOut | None:
    if windows_cannot_open(PurePosixPath(folder).name, walk.reach.platform):
        return WhyLeftOut.UNOPENABLE
    entry = _PathEntry(root / folder)
    if entry.is_junction() or _links_to_a_folder(entry):
        return WhyLeftOut.LINKED_FOLDER
    if _left_out(walk, folder, folder=True):
        return WhyLeftOut.RULES
    return None if not entry.is_dir() or _folder_taken(entry) else WhyLeftOut.PASSED_OVER


def scan_one(root: Path, relative: str, reach: Reach = WITHIN, *, links: LinkChecks = LINK_CHECKS) -> Scanned:
    seen, guard = _Seen(why={}), None if reach.follow_links else _Guard(root.resolve(), links)
    walk = _Walk(seen, [], guard, reach)
    *folders, _ = relative.split("/")
    within = ["/".join(folders[: at + 1]) for at in range(len(folders))]
    passed_over = next((why for folder in within if (why := _passed_over(root, folder, walk)) is not None), None)
    if passed_over is not None:
        _because(seen, relative, passed_over)
        return _scanned(seen)
    entry = _PathEntry(root / relative)
    _taken(entry, "".join(f"{folder}/" for folder in folders), walk)
    if entry.is_dir():
        _because(seen, relative, WhyLeftOut.FOLDER)
    return _scanned(seen)


def scan(
    root: Path,
    reach: Reach = WITHIN,
    *,
    listing: Listing = os.scandir,
    links: LinkChecks = LINK_CHECKS,
    progress: Progress = lambda _: None,
) -> Scanned:
    seen, guard = _Seen(), None if reach.follow_links else _Guard(root.resolve(), links)
    folders, looked_at = [(str(root), "")], 0
    walk = _Walk(seen, folders, guard, reach)
    while folders:
        folder, relative = folders.pop()
        for entry in _listed(folder, relative, listing, seen):
            _taken(entry, relative, walk)
            looked_at += 1
            if looked_at % PROGRESS_EVERY == 0:
                progress(len(seen.files))
    return _scanned(seen)


def _scanned(seen: _Seen) -> Scanned:
    found = Found(
        seen.files,
        tuple(seen.unlisted_folders),
        tuple(seen.unlisted_files),
        tuple(seen.outside),
        tuple(seen.empty_folders),
        seen.left_out,
    )
    return Scanned(
        found,
        tuple(seen.unnamed),
        MappingProxyType(seen.refusals),
        MappingProxyType(seen.identities),
        tuple(seen.linked_folders),
        tuple(seen.named_as_records),
        tuple(seen.looped),
        MappingProxyType(seen.why or {}),
    )
