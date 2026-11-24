import sqlite3
from collections.abc import Generator, Mapping, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn, cast

from sift.api.errors import Problem, StoreUnavailableError
from sift.api.ids import CollectionId, first_row
from sift.domain.codec.shape import SHAPE, Shape
from sift.domain.errors import DamagedError
from sift.storage.database import (
    DAMAGED,
    Reading,
    Value,
    connect,
    first,
    primary_code,
    private_file,
    private_folder,
    reading,
    run_each,
    uri,
    writing,
)
from sift.storage.keyword_meta import keyword_meta
from sift.storage.migrations import Migration, migrated, newer, version
from sift.storage.texts import BLOCK, texts_cut

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
    (
        "ALTER TABLE counts ADD COLUMN vectors INTEGER NOT NULL DEFAULT 0 CHECK (vectors >= 0)",
        "ALTER TABLE counts ADD COLUMN vector_changes INTEGER NOT NULL DEFAULT 0",
        """CREATE TABLE vectors(
            chunk INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            vector BLOB NOT NULL
        ) STRICT""",
        """CREATE TABLE codes(
            chunk INTEGER PRIMARY KEY REFERENCES vectors(chunk) ON DELETE CASCADE,
            bits BLOB NOT NULL
        ) STRICT""",
        """CREATE TRIGGER vectors_added AFTER INSERT ON vectors
           BEGIN UPDATE counts SET vectors = vectors + 1, vector_changes = vector_changes + 1; END""",
        """CREATE TRIGGER vectors_deleted AFTER DELETE ON vectors
           BEGIN UPDATE counts SET vectors = vectors - 1, vector_changes = vector_changes + 1; END""",
        """CREATE TRIGGER vectors_replaced AFTER UPDATE OF vector ON vectors
           BEGIN UPDATE counts SET vector_changes = vector_changes + 1; END""",
    ),
    (
        "CREATE TABLE unembedded(chunk INTEGER PRIMARY KEY) STRICT",
        "INSERT INTO unembedded(chunk) SELECT id FROM chunks WHERE id NOT IN (SELECT chunk FROM vectors)",
        """CREATE TRIGGER chunks_unembedded AFTER INSERT ON chunks
           BEGIN INSERT INTO unembedded(chunk) VALUES (NEW.id); END""",
        """CREATE TRIGGER chunks_gone_unembedded AFTER DELETE ON chunks
           BEGIN DELETE FROM unembedded WHERE chunk = OLD.id; END""",
        """CREATE TRIGGER vectors_embedded AFTER INSERT ON vectors
           BEGIN DELETE FROM unembedded WHERE chunk = NEW.chunk; END""",
        """CREATE TRIGGER vectors_forgotten AFTER DELETE ON vectors
           WHEN EXISTS (SELECT 1 FROM chunks WHERE id = OLD.chunk)
           BEGIN INSERT OR IGNORE INTO unembedded(chunk) VALUES (OLD.chunk); END""",
    ),
    ("ALTER TABLE keyword_segments DROP COLUMN tokens",),
    (
        f"""CREATE TABLE texts(
            document INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            block INTEGER NOT NULL CHECK (block >= 0),
            first_line INTEGER NOT NULL CHECK (first_line >= 1),
            text TEXT NOT NULL CHECK (length(text) BETWEEN 1 AND {BLOCK}),
            PRIMARY KEY(document, block)
        ) STRICT""",
        texts_cut,
        "DROP VIEW keyword_documents",
        "ALTER TABLE documents DROP COLUMN text",
        """CREATE VIEW keyword_documents AS
           SELECT d.id AS id, d.title AS title,
                  (SELECT group_concat(j.atom, ' ') FROM json_tree(d.fields) j
                   WHERE j.atom IS NOT NULL AND j.fullkey <> '$.title') AS meta
           FROM documents d""",
    ),
    ("ALTER TABLE unreadable ADD COLUMN skipped_under TEXT CHECK (skipped_under IN ('off', 'auto'))",),
    (
        "ALTER TABLE chunks ADD COLUMN place INTEGER",
        "UPDATE chunks SET place = id",
        "CREATE UNIQUE INDEX chunks_place ON chunks(place)",
        "ALTER TABLE counts ADD COLUMN places INTEGER NOT NULL DEFAULT 0 CHECK (places >= 0)",
        "UPDATE counts SET places = coalesce((SELECT seq FROM sqlite_sequence WHERE name = 'chunks'), 0)",
        """CREATE TRIGGER chunks_placed AFTER INSERT ON chunks WHEN NEW.place IS NULL
           BEGIN UPDATE counts SET places = max(places + 1, NEW.id);
                 UPDATE chunks SET place = (SELECT places FROM counts) WHERE id = NEW.id; END""",
        """CREATE TRIGGER chunks_given_places AFTER INSERT ON chunks WHEN NEW.place IS NOT NULL
           BEGIN UPDATE counts SET places = max(places, NEW.place); END""",
        "ALTER TABLE keyword_dead RENAME COLUMN chunk TO place",
        "ALTER TABLE counts ADD COLUMN vectors_written INTEGER NOT NULL DEFAULT 0",
        "UPDATE counts SET vectors_written = vectors",
        """CREATE TRIGGER vectors_written AFTER INSERT ON vectors
           BEGIN UPDATE counts SET vectors_written = vectors_written + 1; END""",
        """CREATE TRIGGER vectors_rewritten AFTER UPDATE OF vector ON vectors
           BEGIN UPDATE counts SET vectors_written = vectors_written + 1; END""",
        "ALTER TABLE counts ADD COLUMN ordered_at INTEGER NOT NULL DEFAULT 0",
    ),
    ("ALTER TABLE unreadable ADD COLUMN stopped_under TEXT",),
)
NUMBERED_TABLES = ("documents", "chunks")
FILE_SUFFIXES = ("", "-wal", "-shm")
DATA_SUFFIXES = ("", "-wal")
MISSING_REMEDY = "connect the drive it lies on, or remove the collection"
DAMAGED_REMEDY = "remove the collection, then add its folder and index it again"
BUSY = ("another sift is writing to the store {path}", "wait for it to finish, then run sift again")
NOT_WRITABLE = (
    "the store {path} cannot be written: {error}",
    "see that its drive is connected and writable and that the file is not read-only",
)
UNUSABLE: Mapping[int, tuple[str, str]] = MappingProxyType(
    {
        sqlite3.SQLITE_BUSY: BUSY,
        sqlite3.SQLITE_LOCKED: BUSY,
        sqlite3.SQLITE_FULL: ("the disk holding the store {path} is full", "free space on it, then run sift again"),
        sqlite3.SQLITE_READONLY: NOT_WRITABLE,
        sqlite3.SQLITE_CANTOPEN: NOT_WRITABLE,
        sqlite3.SQLITE_PERM: NOT_WRITABLE,
        sqlite3.SQLITE_IOERR: ("the store {path} could not be read or written: {error}", "check the drive it lies on"),
    }
)


def _files(path: Path, suffixes: tuple[str, ...]) -> list[Path]:
    return [file for file in (Path(f"{path}{suffix}") for suffix in suffixes) if file.is_file()]


LEFT_BY_ADD = """SELECT (SELECT value FROM meta WHERE key = 'collection') = ?
                        AND NOT EXISTS (SELECT 1 FROM documents)"""


def left_by_add(path: Path, collection: CollectionId) -> bool:
    try:
        with closing(sqlite3.connect(uri(path.resolve(), "ro"), uri=True)) as connection:
            (left,) = cast("tuple[int | None]", connection.execute(LEFT_BY_ADD, (str(collection),)).fetchone())
    except sqlite3.DatabaseError:
        return False
    return left == 1


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
    private_folder(path.parent)
    private_file(path)
    try:
        _made(path, collection, layout, shape)
    except BaseException:
        delete_store(path)
        raise


def _damaged(path: Path, error: Exception) -> StoreUnavailableError:
    return StoreUnavailableError(f"the store {path} is damaged: {error}", DAMAGED_REMEDY, code=Problem.STORE_DAMAGED)


def _raise_for(path: Path, error: sqlite3.DatabaseError) -> NoReturn:
    code = primary_code(error)
    if code in DAMAGED:
        raise _damaged(path, error) from None
    if code in UNUSABLE:
        said, remedy = UNUSABLE[code]
        raise StoreUnavailableError(said.format(path=path, error=error), remedy) from None
    raise error


@contextmanager
def unavailable_if_unusable(path: Path) -> Generator[None]:
    try:
        yield
    except sqlite3.DatabaseError as error:
        _raise_for(path, error)
    except DamagedError as error:
        raise _damaged(path, error) from None


def _checked(connection: sqlite3.Connection, path: Path, collection: CollectionId, layout: Sequence[Migration]) -> None:
    with reading(connection) as transaction:
        found = version(transaction)
        if found == 0:
            raise StoreUnavailableError(f"{path} is no store of sift", DAMAGED_REMEDY, code=Problem.STORE_DAMAGED)
        if found > len(layout):
            raise newer(path, found, len(layout))
        held = first(transaction, Value, "SELECT value FROM meta WHERE key = 'collection'")
    if held is None or held.value != str(collection):
        owner = "no collection" if held is None else f"collection {held.value}"
        said = f"{path} is the store of {owner}, not of collection {collection}"
        raise StoreUnavailableError(said, DAMAGED_REMEDY, code=Problem.STORE_DAMAGED)
    if found < len(layout):
        with writing(connection) as transaction:
            migrated(transaction, layout, path)


def open_store(path: Path, collection: CollectionId, layout: Sequence[Migration] = LAYOUT) -> sqlite3.Connection:
    try:
        connection = connect(path, create=False)
    except FileNotFoundError:
        raise StoreUnavailableError(
            f"the store is missing: {path}", MISSING_REMEDY, code=Problem.STORE_MISSING
        ) from None
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


@contextmanager
def connected(path: Path, collection: CollectionId) -> Generator[sqlite3.Connection]:
    with unavailable_if_unusable(path), closing(open_store(path, collection)) as connection:
        yield connection


@contextmanager
def read_store(path: Path, collection: CollectionId) -> Generator[Reading]:
    with connected(path, collection) as connection, reading(connection) as transaction:
        yield transaction
