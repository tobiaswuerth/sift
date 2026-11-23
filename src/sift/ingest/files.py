import os
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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


@contextmanager
def opened(path: Path, scanned: Identity | None) -> Generator[BinaryIO]:
    with path.open("rb") as raw:
        found = os.fstat(raw.fileno())
        if scanned is not None and Identity(found.st_dev, found.st_ino) != scanned:
            raise ChangedError(CHANGED)
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
