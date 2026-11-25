import json
import sqlite3
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple, cast

from sift.api.errors import StoreUnavailableError
from sift.domain.errors import DamagedError
from sift.storage.database import read_only
from sift.storage.store_files import FORMER_CATALOG
from sift.storage.unusable import Unusable, unavailable_if_unusable

MODEL_KEY = "model"
MODEL_AT_KEY = "model_at"
UNPINNED = MappingProxyType(
    {"google/embeddinggemma-2": "google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e"}
)
STORED_TEXT_END = "\x00"
OLDER = "its index is of a sift older than 1.8, which held every collection in one file"
OLDER_REMEDY = "open it once with sift 1.15, which splits it into a store per collection, then import it"
UNREAD_REMEDY = "check that this user may read the sift 1.x home and its stores"
CATALOG_OF_1 = Unusable(
    "the catalog of sift 1.x", "add the folders of its collections to this sift, then index and embed them"
)
STORE_OF_1 = Unusable(
    "the store of sift 1.x",
    "embed the collection: its files are indexed here, and its vectors were in that store alone",
)
CATALOG_COLUMNS = frozenset({"name", "path", "fields", "store", "active"})
STORE_TABLES = frozenset({"docs", "chunks", "vectors", "meta"})
COLUMNS = "SELECT name FROM pragma_table_info('collections')"
TABLES = "SELECT name FROM sqlite_master WHERE type = 'table'"
COLLECTIONS = "SELECT name, path, fields, store, active FROM collections ORDER BY id"
META = "SELECT value FROM meta WHERE key = ?"
COUNTS = "SELECT (SELECT count(*) FROM docs), (SELECT count(*) FROM chunks), (SELECT count(*) FROM vectors)"
DOCUMENTS = "SELECT id, title, text FROM docs ORDER BY id"
EMBEDDED = "SELECT c.id, c.doc, c.seq, c.pos, c.len FROM chunks c JOIN vectors v ON v.chunk = c.id ORDER BY c.doc"
VECTORS = "SELECT chunk, vec FROM vectors WHERE chunk IN (SELECT value FROM json_each(?))"


@dataclass(frozen=True, slots=True)
class FormerCollection:
    name: str
    folder: str
    choices: Mapping[str, object]
    store: Path
    active: bool


@dataclass(frozen=True, slots=True)
class FormerCounts:
    documents: int
    passages: int
    vectors: int


class FormerPassage(NamedTuple):
    chunk: int
    first: bool
    title: str
    text: str


class _FormerSpan(NamedTuple):
    chunk: int
    document: int
    seq: int
    start: int
    length: int


def _opened(path: Path) -> sqlite3.Connection:
    try:
        return read_only(path)
    except sqlite3.Error as error:
        raise StoreUnavailableError(f"{path} cannot be read: {error}", UNREAD_REMEDY) from None


def _names(connection: sqlite3.Connection, sql: str) -> frozenset[str]:
    return frozenset(
        cast("str", row[0]) for row in cast("list[tuple[object, ...]]", connection.execute(sql).fetchall())
    )


def _choices(fields: str | None) -> Mapping[str, object]:
    try:
        read = cast("object", json.loads(fields)) if fields else None
    except ValueError as error:
        raise DamagedError(f"the choices of a collection are no JSON: {error}") from None
    return MappingProxyType(cast("dict[str, object]", read) if isinstance(read, dict) else {})


def former_collections(home: Path) -> tuple[FormerCollection, ...]:
    path = home / FORMER_CATALOG
    with unavailable_if_unusable(path, CATALOG_OF_1), closing(_opened(path)) as connection:
        if not _names(connection, COLUMNS) >= CATALOG_COLUMNS:
            raise StoreUnavailableError(f"{home}: {OLDER}", OLDER_REMEDY)
        found = cast("list[tuple[str, str, str | None, str, int]]", connection.execute(COLLECTIONS).fetchall())
        return tuple(
            FormerCollection(name, folder, _choices(fields), Path(store), bool(active))
            for name, folder, fields, store, active in found
        )


def _cut(text: str) -> str:
    end = text.find(STORED_TEXT_END)
    return text if end < 0 else text[:end]


@dataclass(frozen=True, slots=True)
class FormerStore:
    connection: sqlite3.Connection

    def _meta(self, key: str) -> str | None:
        found = cast("tuple[str | None] | None", self.connection.execute(META, (key,)).fetchone())
        return None if found is None else found[0]

    def model(self) -> str | None:
        name, at = self._meta(MODEL_KEY), self._meta(MODEL_AT_KEY)
        if name is None:
            return None
        return at if at is not None and at.split("@", 1)[0] == name else UNPINNED.get(name, name)

    def counts(self) -> FormerCounts:
        return FormerCounts(*cast("tuple[int, int, int]", self.connection.execute(COUNTS).fetchone()))

    def passages(self) -> Iterator[FormerPassage]:
        documents, held = self.connection.execute(DOCUMENTS), None
        for span in map(
            _FormerSpan._make, cast("Iterator[tuple[int, int, int, int, int]]", self.connection.execute(EMBEDDED))
        ):
            while held is None or held[0] < span.document:
                read = cast("tuple[int, str | None, str] | None", documents.fetchone())
                if read is None:
                    return
                held = (read[0], read[1] or "", _cut(read[2]))
            if held[0] == span.document:
                text = held[2][span.start : span.start + span.length]
                yield FormerPassage(span.chunk, span.seq == 0, held[1], text)

    def vectors(self, chunks: Sequence[int]) -> Mapping[int, bytes]:
        found = cast(
            "list[tuple[int, bytes]]", self.connection.execute(VECTORS, (json.dumps(list(chunks)),)).fetchall()
        )
        return MappingProxyType(dict(found))


@contextmanager
def former_store(path: Path) -> Generator[FormerStore]:
    with unavailable_if_unusable(path, STORE_OF_1), closing(_opened(path)) as connection:
        if not _names(connection, TABLES) >= STORE_TABLES:
            raise StoreUnavailableError(f"{path}: {OLDER}", OLDER_REMEDY)
        yield FormerStore(connection)
