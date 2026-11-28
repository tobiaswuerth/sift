import os
import stat
import sys
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import BinaryIO

READ_AT_ONCE = 1 << 20
CHANGED = "changed while it was indexed; it is read again at the next index"


class ChangedError(OSError):
    pass


@dataclass(frozen=True, slots=True)
class Identity:
    device: int
    inode: int


def identity_at(path: Path) -> Identity:
    found = path.stat()
    return Identity(found.st_dev, found.st_ino)


def own_identity_at(path: Path) -> Identity:
    found = path.stat(follow_symlinks=False)
    return Identity(found.st_dev, found.st_ino)


def _same(path: Path, wanted: Identity) -> bool:
    try:
        return own_identity_at(path) == wanted
    except OSError:
        return False


@dataclass(slots=True)
class Listings:
    held: dict[Path, tuple[frozenset[str], tuple[str, ...]]] = field(default_factory=dict)

    def _listed(self, folder: Path) -> tuple[frozenset[str], tuple[str, ...]]:
        if folder not in self.held:
            try:
                names = tuple(entry.name for entry in folder.iterdir())
            except OSError:
                names = ()
            self.held[folder] = (frozenset(names), names)
        return self.held[folder]

    def _entry(self, folder: Path, part: str) -> str:
        known, names = self._listed(folder)
        if part in known:
            return part
        try:
            wanted = own_identity_at(folder / part)
        except (OSError, ValueError):
            return part
        return next((name for name in names if _same(folder / name, wanted)), part)

    def listed_as(self, root: Path, relative: str) -> str:
        named, folder = list[str](), root
        for part in relative.split("/"):
            listed = self._entry(folder, part)
            named.append(listed)
            folder /= listed
        return "/".join(named)


def _reading() -> int:
    return os.O_RDONLY | (os.O_BINARY if sys.platform == "win32" else os.O_NONBLOCK)


def _changed(found: os.stat_result, scanned: Identity | None) -> bool:
    return not stat.S_ISREG(found.st_mode) or (scanned is not None and Identity(found.st_dev, found.st_ino) != scanned)


def _unchanged(held: int, scanned: Identity | None) -> int:
    if _changed(os.fstat(held), scanned):
        raise ChangedError(CHANGED)
    return held


def _file(held: int, scanned: Identity | None) -> BinaryIO:
    try:
        return os.fdopen(_unchanged(held, scanned), "rb")
    except BaseException:
        os.close(held)
        raise


@contextmanager
def opened(path: Path, scanned: Identity | None) -> Generator[BinaryIO]:
    with _file(os.open(path, _reading()), scanned) as raw:
        yield raw


@contextmanager
def pieces_of(path: Path, scanned: Identity | None) -> Generator[Iterator[bytes]]:
    with opened(path, scanned) as raw:
        yield iter(partial(raw.read, READ_AT_ONCE), b"")


def at_most(path: Path, most: int, scanned: Identity | None) -> bytes:
    read: list[bytes] = []
    left = most + 1
    with opened(path, scanned) as raw:
        while left and (part := raw.read(min(left, READ_AT_ONCE))):
            read.append(part)
            left -= len(part)
    return b"".join(read)
