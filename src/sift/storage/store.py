import sqlite3
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, NoReturn

from sift.api.errors import StoreUnavailableError
from sift.api.ids import CollectionId, first_row
from sift.domain.codec.postings import DamagedError
from sift.domain.codec.segments import SHAPE, Shape
from sift.storage.database import DAMAGED, connect, first, primary_code, reading, run_each, writing
from sift.storage.keywords import keyword_meta
from sift.storage.migrations import Migration, migrated, newer, version

LAYOUT: tuple[Migration, ...] = (
    (
        "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT, WITHOUT ROWID",
        """CREATE TABLE sources(
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            modified_ns INTEGER NOT NULL,
            size INTEGER NOT NULL,
            name TEXT NOT NULL GENERATED ALWAYS AS (substr(path, length(rtrim(path, replace(path, '/', ''))) + 1))
        ) STRICT""",
        "CREATE INDEX sources_name ON sources(name COLLATE NOCASE)",
        """CREATE TABLE unreadable(
            path TEXT PRIMARY KEY,
            problem TEXT NOT NULL,
            read_again INTEGER NOT NULL CHECK (read_again IN (0, 1))
        ) STRICT, WITHOUT ROWID""",
        """CREATE TABLE documents(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            digest TEXT NOT NULL,
            title TEXT NOT NULL,
            fields TEXT NOT NULL CHECK (json_valid(fields)),
            text TEXT NOT NULL,
            UNIQUE(source, key)
        ) STRICT""",
        """CREATE TABLE chunks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            first_line INTEGER NOT NULL,
            last_line INTEGER NOT NULL,
            start INTEGER NOT NULL,
            length INTEGER NOT NULL
        ) STRICT""",
        "CREATE INDEX chunks_document ON chunks(document)",
        """CREATE TABLE links(
            document INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            field TEXT,
            line INTEGER NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('wikilink', 'embed', 'markdown')),
            target TEXT NOT NULL,
            key TEXT NOT NULL,
            fragment TEXT,
            shown TEXT,
            PRIMARY KEY(document, ordinal)
        ) STRICT, WITHOUT ROWID""",
        "CREATE INDEX links_key ON links(key)",
        """CREATE TABLE names(
            key TEXT NOT NULL,
            document INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            kind TEXT NOT NULL CHECK (kind IN ('name', 'alias')),
            PRIMARY KEY(key, document, kind)
        ) STRICT, WITHOUT ROWID""",
        "CREATE INDEX names_document ON names(document)",
        """CREATE TABLE counts(
            id INTEGER PRIMARY KEY CHECK (id = 1),
            documents INTEGER NOT NULL CHECK (documents >= 0),
            chunks INTEGER NOT NULL CHECK (chunks >= 0)
        ) STRICT""",
        "INSERT INTO counts(id, documents, chunks) VALUES (1, 0, 0)",
        """CREATE TRIGGER documents_added AFTER INSERT ON documents
           BEGIN UPDATE counts SET documents = documents + 1; END""",
        """CREATE TRIGGER documents_deleted AFTER DELETE ON documents
           BEGIN UPDATE counts SET documents = documents - 1; END""",
        """CREATE TRIGGER chunks_added AFTER INSERT ON chunks
           BEGIN UPDATE counts SET chunks = chunks + 1; END""",
        """CREATE TRIGGER chunks_deleted AFTER DELETE ON chunks
           BEGIN UPDATE counts SET chunks = chunks - 1; END""",
    ),
    (
        "ALTER TABLE counts ADD COLUMN tokens INTEGER NOT NULL DEFAULT 0 CHECK (tokens >= 0)",
        """CREATE TABLE keyword_segments(
            id INTEGER PRIMARY KEY,
            first INTEGER NOT NULL,
            last INTEGER NOT NULL CHECK (last >= first),
            chunks INTEGER NOT NULL CHECK (chunks > 0),
            dead INTEGER NOT NULL CHECK (dead BETWEEN 0 AND chunks),
            tokens INTEGER NOT NULL CHECK (tokens >= 0),
            average REAL NOT NULL CHECK (average >= 0)
        ) STRICT""",
        """CREATE TABLE keyword_postings(
            segment INTEGER NOT NULL REFERENCES keyword_segments(id) ON DELETE CASCADE,
            term TEXT NOT NULL,
            chunks INTEGER NOT NULL CHECK (chunks > 0),
            postings BLOB,
            positions BLOB,
            ranges BLOB,
            CHECK ((postings IS NULL) = (positions IS NULL) AND (postings IS NULL) <> (ranges IS NULL)),
            PRIMARY KEY(segment, term)
        ) STRICT, WITHOUT ROWID""",
        """CREATE TABLE keyword_blocks(
            segment INTEGER NOT NULL,
            term TEXT NOT NULL,
            number INTEGER NOT NULL CHECK (number >= 0),
            postings BLOB NOT NULL,
            positions BLOB NOT NULL,
            PRIMARY KEY(segment, term, number),
            FOREIGN KEY(segment, term) REFERENCES keyword_postings(segment, term) ON DELETE CASCADE
        ) STRICT, WITHOUT ROWID""",
        """CREATE TABLE keyword_lengths(
            segment INTEGER NOT NULL REFERENCES keyword_segments(id) ON DELETE CASCADE,
            block INTEGER NOT NULL,
            lengths BLOB NOT NULL,
            PRIMARY KEY(segment, block)
        ) STRICT, WITHOUT ROWID""",
        """CREATE TABLE keyword_terms(
            term TEXT PRIMARY KEY,
            chunks INTEGER NOT NULL CHECK (chunks > 0)
        ) STRICT, WITHOUT ROWID""",
        "CREATE TABLE keyword_dead(chunk INTEGER PRIMARY KEY) STRICT",
        """CREATE VIEW keyword_documents AS
           SELECT d.id AS id, d.title AS title, d.text AS text,
                  (SELECT group_concat(j.atom, ' ') FROM json_tree(d.fields) j
                   WHERE j.atom IS NOT NULL AND j.fullkey <> '$.title') AS meta
           FROM documents d""",
        """CREATE TRIGGER keyword_segment_dropped AFTER DELETE ON keyword_segments
           BEGIN DELETE FROM keyword_dead WHERE chunk BETWEEN OLD.first AND OLD.last; END""",
    ),
)
NUMBERED_TABLES = ("documents", "chunks")
FILE_SUFFIXES = ("", "-wal", "-shm")
DATA_SUFFIXES = ("", "-wal")
MISSING_REMEDY = "connect the drive it lies on, or remove the collection"
DAMAGED_REMEDY = "remove the collection, then add its folder and index it again"


class Held(NamedTuple):
    value: str


def _files(path: Path, suffixes: tuple[str, ...]) -> list[Path]:
    return [file for file in (Path(f"{path}{suffix}") for suffix in suffixes) if file.is_file()]


def delete_store(path: Path) -> None:
    for file in _files(path, FILE_SUFFIXES):
        file.unlink()


def _size(file: Path) -> int:
    try:
        return file.stat().st_size
    except FileNotFoundError:
        return 0


def stored_bytes(path: Path) -> int:
    return sum(_size(Path(f"{path}{suffix}")) for suffix in DATA_SUFFIXES)


def _made(path: Path, collection: CollectionId, layout: Sequence[Migration], shape: Shape) -> None:
    connection = connect(path, create=False)
    try:
        with writing(connection) as transaction:
            migrated(transaction, layout, path)
            start = first_row(collection)
            run_each(
                transaction,
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                [("collection", str(collection)), *keyword_meta(transaction, shape, start)],
            )
            run_each(
                transaction,
                "INSERT INTO sqlite_sequence(name, seq) VALUES (?, ?)",
                [(table, start) for table in NUMBERED_TABLES],
            )
    finally:
        connection.close()


def create_store(
    path: Path, collection: CollectionId, layout: Sequence[Migration] = LAYOUT, shape: Shape = SHAPE
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.open("xb").close()
    try:
        _made(path, collection, layout, shape)
    except BaseException:
        delete_store(path)
        raise


def _raise_for(path: Path, error: sqlite3.DatabaseError) -> NoReturn:
    if primary_code(error) in DAMAGED:
        raise StoreUnavailableError(f"the store {path} is damaged: {error}", DAMAGED_REMEDY) from None
    raise error


@contextmanager
def unavailable_if_damaged(path: Path) -> Generator[None]:
    try:
        yield
    except sqlite3.DatabaseError as error:
        _raise_for(path, error)
    except DamagedError as error:
        raise StoreUnavailableError(f"the store {path} is damaged: {error}", DAMAGED_REMEDY) from None


def _checked(connection: sqlite3.Connection, path: Path, collection: CollectionId, layout: Sequence[Migration]) -> None:
    with reading(connection) as transaction:
        found = version(transaction)
        if found == 0:
            raise StoreUnavailableError(f"{path} is no store of sift", DAMAGED_REMEDY)
        if found > len(layout):
            raise newer(path, found, len(layout))
        held = first(transaction, Held, "SELECT value FROM meta WHERE key = 'collection'")
    if held is None or held.value != str(collection):
        owner = "no collection" if held is None else f"collection {held.value}"
        raise StoreUnavailableError(f"{path} is the store of {owner}, not of collection {collection}", DAMAGED_REMEDY)
    if found < len(layout):
        with writing(connection) as transaction:
            migrated(transaction, layout, path)


def open_store(path: Path, collection: CollectionId, layout: Sequence[Migration] = LAYOUT) -> sqlite3.Connection:
    try:
        connection = connect(path, create=False)
    except FileNotFoundError:
        raise StoreUnavailableError(f"the store is missing: {path}", MISSING_REMEDY) from None
    except sqlite3.DatabaseError as error:
        _raise_for(path, error)
    try:
        _checked(connection, path, collection, layout)
    except BaseException as error:
        connection.close()
        if isinstance(error, sqlite3.DatabaseError):
            _raise_for(path, error)
        raise
    return connection
