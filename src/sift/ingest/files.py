import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

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


@contextmanager
def opened(path: Path, scanned: Identity | None) -> Generator[BinaryIO]:
    with path.open("rb") as raw:
        found = os.fstat(raw.fileno())
        if scanned is not None and Identity(found.st_dev, found.st_ino) != scanned:
            raise ChangedError(CHANGED)
        yield raw


def at_most(path: Path, most: int, scanned: Identity | None) -> bytes:
    with opened(path, scanned) as raw:
        return raw.read(most + 1)
