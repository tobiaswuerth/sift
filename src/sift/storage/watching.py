import itertools
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from sift.storage.catalog import CATALOG, known_collections
from sift.storage.database import read_only

type Seen = tuple[int | None, ...]


def _unknown() -> Iterator[int]:
    return itertools.count(-1, -1)


@dataclass(slots=True, eq=False)
class HomeWatch:
    home: Path
    held: dict[Path, sqlite3.Connection] = field(default_factory=dict[Path, sqlite3.Connection])
    stores: tuple[Path, ...] = ()
    catalog: int | None = None
    listed: bool = False
    unknown: Iterator[int] = field(default_factory=_unknown)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def seen(self) -> Seen:
        with self.lock:
            catalog = self._version(self.home / CATALOG)
            if not self.listed or catalog != self.catalog:
                self.stores = tuple(sorted(one.store for one in known_collections(self.home) if one.active))
                self.catalog, self.listed = catalog, True
                for path in set(self.held) - {self.home / CATALOG, *self.stores}:
                    self.held.pop(path).close()
            return (catalog, *map(self._version, self.stores))

    def let_go(self) -> None:
        with self.lock:
            for connection in self.held.values():
                connection.close()
            self.held.clear()
            self.listed = False

    def _version(self, path: Path) -> int | None:
        if not path.exists():
            return None
        try:
            if path not in self.held:
                self.held[path] = read_only(path)
            (version,) = cast("tuple[int]", self.held[path].execute("PRAGMA data_version").fetchone())
        except sqlite3.Error:
            if path in self.held:
                self.held.pop(path).close()
            return next(self.unknown)
        return version
