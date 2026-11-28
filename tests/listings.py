import os
import stat
from collections.abc import Generator, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path

from sift.ingest.files import Identity
from sift.ingest.scan import DirEntry, LinkChecks, Listing, Scanned


@dataclass(frozen=True, slots=True)
class Fake:
    name: str
    path: str
    folder: bool = False
    junction: bool = False
    symlink: bool = False
    mode: int = stat.S_IFREG | 0o644
    modified_ns: int = 7
    refusal: OSError | None = None
    untyped: OSError | None = None
    unfollowed: OSError | None = None
    attributes: int = 0

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        if self.untyped is not None:
            raise self.untyped
        if follow_symlinks and self.unfollowed is not None:
            raise self.unfollowed
        return self.folder and (follow_symlinks or not self.symlink)

    def is_junction(self) -> bool:
        return self.junction

    def is_symlink(self) -> bool:
        return self.symlink

    def stat(self) -> os.stat_result:
        if self.refusal is not None:
            raise self.refusal
        given = {"st_mtime_ns": self.modified_ns, "st_file_attributes": self.attributes}
        held = {name: value for name, value in given.items() if hasattr(os.stat_result, name)}
        return os.stat_result((self.mode, 0, 0, 0, 0, 0, 3, 0, 0, 0), held)


@dataclass(frozen=True, slots=True)
class Breaking:
    entries: tuple[DirEntry, ...]
    error: OSError

    def __iter__(self) -> Iterator[DirEntry]:
        yield from self.entries
        raise self.error


type Listed = Iterable[DirEntry] | OSError


def links(targets: Mapping[str, str]) -> LinkChecks:
    def target(path: Path) -> str | None:
        return targets.get(str(path))

    def identity(path: Path) -> Identity:
        return Identity(1, len(path.parts))

    return LinkChecks(target, identity, identity)


FAKE = links({})


def listing(folders: Mapping[str, Listed]) -> Listing:
    @contextmanager
    def listed(folder: str) -> Generator[Iterable[DirEntry]]:
        given = folders.get(folder)
        if given is None:
            with os.scandir(folder) as real:
                yield list(real)
            return
        if isinstance(given, OSError):
            raise given
        yield given

    def opened(folder: str) -> AbstractContextManager[Iterable[DirEntry]]:
        given = folders.get(folder)
        if isinstance(given, OSError):
            raise given
        return listed(folder)

    return opened


def paths(scanned: Scanned) -> list[str]:
    return sorted(scanned.found.files)
