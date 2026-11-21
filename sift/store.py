"""A collection's database, its store: its documents and the last parts of their paths, chunks, keyword index (FTS5)
and keyword postings, vectors and their codes, its notes' links, and the counts of its rows, in one SQLite file. The
catalog (see `catalog`) lists the collections and where their stores lie.

A store holds the text of every document, so searching and reading never touch the files; they are read only when
indexing (see `indexing`, and `readers` for the kinds of file).

The ids of a collection's documents and chunks start at its id << SHIFT: unique across collections, they say whose
they are (`owner`), so search results from several stores merge as they are.
"""

import json
import sqlite3
import threading
import uuid
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload
from urllib.parse import quote

from . import console, readers

if TYPE_CHECKING:
    from .vectors import Bits, Floats, Ids

_D = TypeVar("_D")

SCHEMA_VERSION = 10  # in SQLite's user_version; up to sift 1.7, one database held every collection (see `upgrade`)
FIRST_LAYOUT = 6  # sift 1.8's, the first of a store per collection; its rows are counted once (see _count_rows)
# sift 1.9's, which counts rows; 8 adds the keyword postings' tables (POSTINGS), 9 their order, and 10 the last parts
# of the documents' paths (BASENAMES)
COUNTED_LAYOUT = 7
TIMEOUT = 30  # seconds to wait for another sift's transaction
WAL_LIMIT = 64 * 2**20  # bytes the write-ahead log shrinks back to once written into the store: SQLite keeps its size
UPGRADE_WAIT = 3600  # seconds to wait while another sift counts a store's rows, once: minutes for a large one
# While set, every statement on every database stops with "interrupted": a server sets it for a call its client
# cancelled (see `jobs`). SQLite looks every ABORT_STEPS instructions, about a millisecond.
ABORT = threading.Event()
ABORT_STEPS = 100_000
SHIFT = 40  # a collection's row ids start at its id << SHIFT: a trillion rows each, and millions of collections
# The keyword index holds what a view reads of each chunk: its text, its document's title, and its front-matter values
# but the title, e.g. authors and ids, as `meta`. Entries are added and deleted by the same rows (see `fill`), so the
# two always agree: the view's, which for a document of many chunks are cut here instead (chunk_rows), the same.
TOKENIZER = "unicode61 remove_diacritics 2"  # FTS5's, as the keyword postings tokenize too (see `postings`)
META = "(SELECT group_concat(j.atom, ' ') FROM json_tree(d.meta) j WHERE j.atom IS NOT NULL AND j.fullkey != '$.title')"
KEYWORD_INDEX = f"""
CREATE VIEW IF NOT EXISTS chunk_text AS SELECT c.id AS id, d.title AS title, substr(d.text, c.pos + 1, c.len) AS text,
    {META}
    AS meta FROM chunks c JOIN docs d ON d.id = c.doc;
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(title, text, meta, content='chunk_text', content_rowid='id',
    tokenize='{TOKENIZER}');
"""
CUT = 10_000  # chunks cut out of their documents' texts at a time, by chunk_rows
# The chunks from which a document's are cut by chunk_rows, not by the view: the view's substr loads the whole text and
# walks to the position for each chunk, which past some 20 KB of text costs more than cutting it here, and for the
# 32,795 chunks of an 80 MB file took 11 minutes. Below, the view's one statement is quicker by a tenth.
LONG = 8
# AUTOINCREMENT: ids grow from the collection's first (in sqlite_sequence), and are never handed out twice.
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sources(id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, mtime INTEGER, size INTEGER);
CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY AUTOINCREMENT, source INTEGER NOT NULL,
    path TEXT UNIQUE NOT NULL, hash TEXT, title TEXT, meta TEXT, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS docs_source ON docs(source);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY AUTOINCREMENT, doc INTEGER NOT NULL, seq INTEGER,
    line_start INTEGER, line_end INTEGER, pos INTEGER NOT NULL, len INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc);
CREATE TABLE IF NOT EXISTS vectors(chunk INTEGER PRIMARY KEY, vec BLOB NOT NULL);
"""
# The links of Markdown notes as written, and the names files answer to, both by key: see `links`, which resolves the
# one to the other when asked.
LINKS = """
CREATE TABLE IF NOT EXISTS links(id INTEGER PRIMARY KEY, doc INTEGER NOT NULL, field TEXT, line INTEGER NOT NULL,
    kind TEXT NOT NULL, target TEXT NOT NULL, key TEXT NOT NULL, fragment TEXT, shown TEXT);
CREATE INDEX IF NOT EXISTS links_doc ON links(doc);
CREATE INDEX IF NOT EXISTS links_key ON links(key);
CREATE TABLE IF NOT EXISTS names(doc INTEGER NOT NULL, key TEXT NOT NULL, kind TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS names_doc ON names(doc);
CREATE INDEX IF NOT EXISTS names_key ON names(key);
"""
# Each vector's code, its signs a bit per dimension, which searches load instead of the vectors (see `vectors`);
# written with the vector (put_vectors), and deleted with it by the trigger, wherever vectors are deleted.
CODES = """
CREATE TABLE IF NOT EXISTS codes(chunk INTEGER PRIMARY KEY, bits BLOB NOT NULL);
CREATE TRIGGER IF NOT EXISTS codes_go_with_vectors AFTER DELETE ON vectors
    BEGIN DELETE FROM codes WHERE chunk = OLD.chunk; END;
"""
# The rows of each table a status shows, counted as they come and go, since SQLite keeps no counts and counting a large
# store's vectors reads all of them: by triggers, in the transaction of the change itself, so that no interruption
# separates rows and counts. Vectors are written as upserts: SQLite deletes the row an INSERT OR REPLACE replaces
# without firing delete triggers.
COUNTED = ("docs", "chunks", "vectors")
COUNTS = (
    "CREATE TABLE IF NOT EXISTS counts(name TEXT PRIMARY KEY, n INTEGER NOT NULL)",
    *(
        f"CREATE TRIGGER IF NOT EXISTS {table}_{change} AFTER {event} ON {table} "
        f"BEGIN UPDATE counts SET n = n {sign} 1 WHERE name = '{table}'; END"
        for table in COUNTED
        for change, event, sign in (("added", "INSERT", "+"), ("removed", "DELETE", "-"))
    ),
)
# The last part of each document's path, a file's name or a record's id: what a document is looked up by when it is
# named otherwise than by its path, e.g. by a shortened id (see search.Index._rows), where every path was read.
# Compared without ASCII case, as an absolute path names its file in another case too. Triggers keep it, in the
# transaction of each change, so that no document is without its row, whoever writes. A store of sift 1.12 or older
# gets the table empty, and `sift index` lists its documents (list_basenames): until it has, meta LISTED is missing.
LAST_PART = "substr({0}, length(rtrim({0}, replace({0}, '/', ''))) + 1)"  # of {0}: rtrim leaves it up to its last "/"
BASENAMES = (
    """CREATE TABLE IF NOT EXISTS basenames(name TEXT NOT NULL COLLATE NOCASE, doc INTEGER NOT NULL,
        PRIMARY KEY(name, doc)) WITHOUT ROWID""",
    f"""CREATE TRIGGER IF NOT EXISTS basenames_added AFTER INSERT ON docs
        BEGIN INSERT INTO basenames(name, doc) VALUES({LAST_PART.format("NEW.path")}, NEW.id); END""",
    f"""CREATE TRIGGER IF NOT EXISTS basenames_removed AFTER DELETE ON docs
        BEGIN DELETE FROM basenames WHERE name = {LAST_PART.format("OLD.path")} AND doc = OLD.id; END""",
    f"""CREATE TRIGGER IF NOT EXISTS basenames_moved AFTER UPDATE OF path ON docs
        BEGIN DELETE FROM basenames WHERE name = {LAST_PART.format("OLD.path")} AND doc = OLD.id;
        INSERT INTO basenames(name, doc) VALUES({LAST_PART.format("NEW.path")}, NEW.id); END""",
)
BASENAMES_BATCH = 50_000  # documents listed per transaction in an older store, which holds it for some 50 ms
LISTED, LISTING = "basenames", "basenames_to"  # in the meta table: every document is listed; how far a listing got
LISTING_START = '[0, ""]'  # no document listed yet, and the path every path follows: none is empty
CODES_BATCH = 100_000  # codes per batch as searches load them, and per transaction as vectors get them afterwards
# Keyword search's own postings (see `postings`), built from FTS5's index, which stays the reference: per term and
# segment its postings, long lists in blocks; chunks' lengths, in blocks of consecutive places; the chunks whose
# postings in older segments are void (masked); the chunk at each place, when the postings number chunks in an order
# of their own (kw_order). Once built (meta kw_built), every chunk whose FTS5 entries change is marked dirty, in the
# same transaction, numbered by the change (meta kw_change), until the postings follow.
POSTINGS = """
CREATE TABLE IF NOT EXISTS kw_segments(id INTEGER PRIMARY KEY, chunks INTEGER NOT NULL, tokens INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS kw_terms(term TEXT NOT NULL, segment INTEGER NOT NULL, df INTEGER NOT NULL, list INTEGER,
    data BLOB, PRIMARY KEY(term, segment)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS kw_blocks(id INTEGER PRIMARY KEY, data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS kw_lengths(segment INTEGER NOT NULL, block INTEGER NOT NULL, data BLOB NOT NULL,
    PRIMARY KEY(segment, block)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS kw_masked(chunk INTEGER PRIMARY KEY, segment INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS kw_dirty(chunk INTEGER PRIMARY KEY, change INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS kw_order(block INTEGER PRIMARY KEY, data BLOB NOT NULL);
"""


def postings_lock(path: str | Path) -> Path:
    """The file beside the store at `path` whose lock is held while its keyword postings change (postings._alone)."""
    return Path(f"{path}.postings-lock")


def files(path: str | Path) -> list[Path]:
    """The files of the store at `path`: the database, its write-ahead log and shared memory, and its postings' lock."""
    return [Path(path), Path(f"{path}-wal"), Path(f"{path}-shm"), postings_lock(path)]


def first_id(cid: int) -> int:
    """The first row id of the collection `cid`."""
    return cid << SHIFT


def owner(row_id: int) -> int:
    """The id of the collection a document or chunk, by its row id, belongs to."""
    return int(row_id) >> SHIFT


class Connection(sqlite3.Connection):
    """A database's connection that knows how many rows it changed in tables of its own (see scratch): they change
    nothing of the database, and `changes` leaves them out."""

    scratch = 0


class Collection(Protocol):
    """What the modules below the catalog ask of its collections (catalog.Collection): catalog imports them, through
    upgrade, so they can't import it."""

    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def path(self) -> str: ...
    @property
    def store(self) -> str: ...
    @property
    def active(self) -> bool: ...
    @property
    def choices(self) -> dict[str, Any]: ...


C = TypeVar("C", bound=Collection)


class Catalog(Protocol[C]):
    """What the modules below the catalog ask of it (catalog.Catalog)."""

    def collections(self) -> list[C]: ...
    def get(self, name: str) -> C: ...
    def store(self, collection: C) -> Connection: ...
    def opened(self, collection: C) -> Connection: ...
    def set_title_from(self, name: str, title_from: str) -> C: ...


@contextmanager
def scratch(db: Connection) -> Generator[None]:
    """While this lasts, `db` writes only tables of its own, temporary ones: the rows it changes are counted as such,
    as SQLite counts them among the connection's changes, where they looked like changes of the store."""
    before = db.total_changes
    try:
        yield
    finally:
        db.scratch += db.total_changes - before


def changes(db: Connection) -> tuple[int, int]:
    """How SQLite counts a database's changes through a connection: those other connections committed, and the rows
    the connection changed itself, but those in tables of its own (see scratch)."""
    return db.execute("PRAGMA data_version").fetchone()[0], db.total_changes - db.scratch


def database(path: str | Path, create: bool = False) -> Connection:
    """The SQLite database at `path`, with sift's settings. It is made only if `create`: a store on a drive that isn't
    connected stays missing, and is never made anew, empty."""
    path = Path(path)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=TIMEOUT, check_same_thread=False, factory=Connection)
    elif not path.is_file():
        raise FileNotFoundError(f"not found: {path}")
    else:
        db = sqlite3.connect(
            uri(path.resolve(), "rw"), timeout=TIMEOUT, check_same_thread=False, factory=Connection, uri=True
        )
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(f"PRAGMA journal_size_limit={WAL_LIMIT}")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("PRAGMA cache_size=-262144")  # 256 MB
        db.set_progress_handler(ABORT.is_set, ABORT_STEPS)
    except BaseException:
        db.close()
        raise
    return db


def read_only(path: str | Path) -> Connection:
    """The database at `path`, to read only, through a connection of its own: no setting changes, and no call
    cancelled meanwhile (ABORT) stops what it reads."""
    return sqlite3.connect(uri(Path(path).resolve(), "ro"), uri=True, timeout=TIMEOUT, factory=Connection)


def uri(path: Path, mode: str) -> str:
    r"""The URI SQLite opens the file at the absolute `path` by, to read and write it ("rw") or to read it only ("ro").
    A Windows network path, \\nas\share\x, as file:////nas/share/x: SQLite refuses a URI that names a host, as
    Path.as_uri writes it (file://nas/share/x), and a drive mapped to a share resolves to its network path. Written
    here, not by as_uri, which Python 3.14 deprecates for a path of no system's own (PurePath)."""
    written = path.as_posix()  # D:/x, //nas/share/x, /home/x
    drive, rest = (written[:2], written[2:]) if written[1:2] == ":" else ("", written)
    return f"file:{'///' if drive else '//'}{drive}{quote(rest)}?mode={mode}"


DAMAGED = (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB)


def damage(name: str, error: BaseException) -> str:
    """What to say of the collection `name` whose store SQLite found damaged as it read it, as a failing disk leaves
    one, and what to do. Any other error of SQLite's (locked, full, misused) is raised again: not the store's fault."""
    if (getattr(error, "sqlite_errorcode", 0) & 0xFF) not in DAMAGED:
        raise error
    forget = console.command("sift", "rm", name)
    return f"{name}: its index is damaged ({error}): `{forget}` forgets it; then add its folder and index it again"


def connect(path: str | Path, cid: int, create: bool = False) -> Connection:
    """The store of the collection `cid` at `path`; with `create`, a new one, made there."""
    db = database(path, create)
    try:
        _layout(db, path, cid, create)
    except BaseException:
        db.close()
        raise
    return db


def _layout(db: sqlite3.Connection, path: str | Path, cid: int, create: bool) -> None:
    """Make a new store's tables, or check that an existing one is this collection's, in this version's layout."""
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise SystemExit(f"{path} was made by a newer sift; upgrade sift to use it")
    if create and not version:
        make(db, cid)
        db.commit()
        return
    meta = db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'").fetchone()
    held = db.execute("SELECT value FROM meta WHERE key = 'collection'").fetchone() if meta else None
    if version < FIRST_LAYOUT or held is None:
        raise SystemExit(f"{path} is not the index of a collection of this sift")
    if int(held[0]) != cid:
        raise SystemExit(f"{path} holds the index of another collection (id {held[0]}, not {cid})")
    if version < COUNTED_LAYOUT:
        _count_rows(db, path)
    if version < SCHEMA_VERSION:
        _add_tables(db)
    for key, kinds in READ_AGAIN.items():
        if get_meta(db, key) is None:
            _once(db, key, partial(read_again, kinds=kinds))
    if get_meta(db, WAITED) is None:
        _once(db, WAITED, _read_waited_again)


def make(db: sqlite3.Connection, cid: int, read: bool = True) -> None:
    """The tables of a new store of the collection `cid`, its ids starting at first_id(cid). Unless `read`, its files
    of the kinds an older sift skipped are still to be read again (READ_AGAIN)."""
    kept_by_triggers = "".join(f"{statement};\n" for statement in (*COUNTS, *BASENAMES))
    db.executescript(SCHEMA + KEYWORD_INDEX + LINKS + CODES + POSTINGS + kept_by_triggers)
    db.executemany("INSERT INTO counts(name, n) VALUES(?, 0)", [(table,) for table in COUNTED])
    first = first_id(cid)
    db.executemany("INSERT INTO sqlite_sequence(name, seq) VALUES(?, ?)", [("docs", first), ("chunks", first)])
    set_meta(db, "collection", cid)
    set_meta(db, LISTED, 1)  # its documents are listed as they come
    for key in (*READ_AGAIN, WAITED) if read else ():
        set_meta(db, key, 1)
    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


@contextmanager
def alone(path: str | Path, name: str) -> Generator[sqlite3.Connection]:
    """The store at `path` of the collection `name`, held so that no other sift reads or writes it until this ends;
    its write-ahead log is written into it first, so that the file holds it all. An error if another sift has it
    open."""
    db = sqlite3.connect(uri(Path(path).resolve(), "rw"), uri=True, timeout=1, isolation_level=None)
    try:
        db.execute("PRAGMA locking_mode=EXCLUSIVE")
        try:
            db.execute("BEGIN EXCLUSIVE")  # this locking mode holds on to the lock after COMMIT
        except sqlite3.OperationalError:
            raise SystemExit(
                f"{name}: another sift has its index open (a server, `sift index` or `sift embed`): stop it, then "
                "try again"
            ) from None
        db.execute("COMMIT")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        yield db
    finally:
        db.close()


READ_AGAIN: dict[
    str, tuple[str, ...]
] = {  # in the meta table, once the files of these kinds that an older sift skipped are to be read again
    "scans_read": (".pdf",),  # sift 1.2 and older read no scans
    "documents_read": readers.DOCUMENTS,  # the server of sift 1.3.0 and older skipped every document it reindexed
}
WAITED = "paths_waited"  # in the meta table, once what sift 1.11 left out for a taken path is to be read again


def _once(db: sqlite3.Connection, key: str, work: Callable[[sqlite3.Connection], object]) -> None:
    """work(db), once per store, as the meta `key` records: in one transaction, which another sift starting at the same
    time waits for, and then finds done. Not in a store this sift may only read: once it may write."""
    try:
        db.execute("BEGIN IMMEDIATE")
        try:
            if get_meta(db, key) is None:
                work(db)
                set_meta(db, key, 1)
            db.commit()
        except BaseException:
            db.rollback()
            raise
    except sqlite3.OperationalError as error:
        if getattr(error, "sqlite_errorcode", 0) & 0xFF != sqlite3.SQLITE_READONLY:
            raise


def _read_waited_again(db: sqlite3.Connection) -> None:
    """Have the next `sift index` read again what sift 1.11 left out for good, as another file's document held its path,
    with nothing to say that it waits (see wait). Every file without documents, as a file left out whole is, whether
    or not its path is free by now; and a JSON Lines file or mail below whose name lie more documents than its own, as
    when one of its records or attachments was left out for a file there. The indexes alone are read, not the texts."""
    db.execute("UPDATE sources SET mtime = NULL WHERE NOT EXISTS (SELECT 1 FROM docs WHERE docs.source = sources.id)")
    kinds = json.dumps(readers.JSONL + readers.EMAIL)  # json_each has a column `path` of its own: sources.path
    holders = db.execute(
        "SELECT id, path FROM sources "
        "WHERE EXISTS (SELECT 1 FROM json_each(?) k WHERE lower(sources.path) LIKE '%' || k.value)",
        (kinds,),
    ).fetchall()
    for sid, path in holders:
        below = (f"{path.rsplit('.', 1)[0]}/", f"{path.rsplit('.', 1)[0]}0")  # '0' follows '/'
        if not db.execute("SELECT 1 FROM sources WHERE path >= ? AND path < ? LIMIT 1", below).fetchone():
            continue  # no file below its name, for whom one of its documents might have been left out
        found = db.execute("SELECT count(*) FROM docs WHERE path >= ? AND path < ?", below).fetchone()[0]
        own = db.execute("SELECT count(*) FROM docs WHERE source = ?", (sid,)).fetchone()[0]
        if found > own - path.lower().endswith(readers.EMAIL):  # a mail's own document lies beside, not below
            db.execute("UPDATE sources SET mtime = NULL WHERE id = ?", (sid,))


def read_again(db: sqlite3.Connection, kinds: tuple[str, ...]) -> None:
    """Have the next `sift index` read again the files of these kinds without documents, and the mails with one
    attached: those an older sift skipped (READ_AGAIN), or scans left unread (indexing.SCANS_SKIPPED). Their
    unchanged documents stay, vectors included."""
    db.execute(
        """UPDATE sources SET mtime = NULL
           WHERE (id NOT IN (SELECT source FROM docs)
                  AND EXISTS (SELECT 1 FROM json_each(:kinds) k WHERE lower(sources.path) LIKE '%' || k.value))
           OR (lower(sources.path) LIKE '%.eml' AND EXISTS (SELECT 1 FROM docs d,
               json_each(CASE WHEN json_type(d.meta, '$.attachments') = 'array'
                         THEN json_extract(d.meta, '$.attachments') END) a, json_each(:kinds) k
               WHERE d.source = sources.id AND lower(a.value) LIKE '%' || k.value))""",
        {"kinds": json.dumps(kinds)},
    )


def _count_rows(db: sqlite3.Connection, path: str | Path) -> None:
    """Count the rows of a store of sift 1.8 once, a pass over each table, and keep them counted from then on (layout
    6 to 7). In one transaction, which another sift opening the store meanwhile waits for, and then finds done."""
    db.execute(f"PRAGMA busy_timeout = {UPGRADE_WAIT * 1000}")
    try:
        db.execute("BEGIN IMMEDIATE")
        try:
            if db.execute("PRAGMA user_version").fetchone()[0] < COUNTED_LAYOUT:
                console.say(f"counting the rows of {path} once, to keep them counted from now on ...")
                for statement in COUNTS:
                    db.execute(statement)
                rows = [(table, db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]) for table in COUNTED]
                db.executemany("INSERT INTO counts(name, n) VALUES(?, ?)", rows)
                db.execute(f"PRAGMA user_version = {COUNTED_LAYOUT}")
            db.commit()
        except BaseException:
            db.rollback()
            raise
    finally:
        db.execute(f"PRAGMA busy_timeout = {TIMEOUT * 1000}")


def _add_tables(db: sqlite3.Connection) -> None:
    """Add the tables a store lacks, in one transaction, which another sift opening the store meanwhile waits for, and
    then finds done. The keyword postings': all of them to a store of sift 1.9 (layout 7), unbuilt, so that keyword
    search reads FTS5 until `sift index` builds them (see `postings`); their order to one of 1.10 (layout 8), empty,
    which keeps its postings' chunks in the order they were indexed. And to a store of 1.12 or older (layout 9), the
    last parts of its documents' paths: empty, kept from now on, and for `sift index` to list (list_basenames)."""
    db.execute("BEGIN IMMEDIATE")
    try:
        if db.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
            # statement by statement: executescript would commit first
            for statement in (*filter(str.strip, POSTINGS.split(";")), *BASENAMES):
                db.execute(statement)
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        db.commit()
    except BaseException:
        db.rollback()
        raise


def basenames_listed(db: sqlite3.Connection) -> bool:
    """Whether the store lists the last part of every document's path (see BASENAMES): one this sift made does, an
    older one once `sift index` has listed them."""
    return get_meta(db, LISTED) is not None


def list_basenames(db: sqlite3.Connection, name: str = "") -> None:
    """List the last part of every document's path in a store of sift 1.12 or older, the collection `name`'s, once: a
    batch of documents per transaction, so that this goes on where it stopped, and another sift's change waits for
    a batch, not for the store. Documents that come and go meanwhile are the triggers'."""
    if basenames_listed(db):
        return
    before = json.loads(get_meta(db, LISTING, LISTING_START))[0]  # what a listing that was stopped has listed
    left = max(counts(db)["docs"] - before, 0)
    progress = console.Progress(left, "documents", prefix=f"{name}: ")  # its clock starts before the first batch
    if (done := _list_batch(db)) is None:  # none to list: nothing to say
        return
    console.say(f"{name}: listing the last part of each document's path, once, to look up shortened ids by ...")
    with progress:
        while done is not None:
            progress.update(done - before, max(done - before, left))
            done = _list_batch(db)


def _list_batch(db: sqlite3.Connection) -> int | None:
    """List the next BASENAMES_BATCH documents in the order of their paths, and record how far this got. Returns how
    many are listed by now; None once none was left, the store marked as listed, by this sift or another meanwhile.
    The documents are read from the index on their paths alone, never with their texts. The batch is found before its
    transaction, which then reads pages the drive has given: it holds the store only for as long as writing the batch
    takes, and other sifts' changes get their turn between two batches."""
    got_to = get_meta(db, LISTING, LISTING_START)
    done, after = json.loads(got_to)
    batch = "SELECT path FROM docs WHERE path > ? ORDER BY path LIMIT ?"
    n, last = db.execute(f"SELECT count(*), max(path) FROM ({batch})", (after, BASENAMES_BATCH)).fetchone()
    db.execute("BEGIN IMMEDIATE")
    try:
        if basenames_listed(db):
            done = None
        elif (now := get_meta(db, LISTING, LISTING_START)) != got_to:  # another sift listed this batch: on after it
            done = json.loads(now)[0]
        elif n:  # as the documents are now: those gone meanwhile get no row, those that came have theirs
            db.execute(
                f"INSERT OR IGNORE INTO basenames(name, doc) SELECT {LAST_PART.format('path')}, id FROM docs "
                "WHERE path > ? AND path <= ?",
                (after, last),
            )
            done += n
            set_meta(db, LISTING, json.dumps([done, last]))
        else:
            db.execute("DELETE FROM meta WHERE key = ?", (LISTING,))
            set_meta(db, LISTED, 1)
            done = None
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return done


def counts(db: sqlite3.Connection) -> dict[str, int]:
    """How many rows of each table in COUNTED the store holds, by table, as kept: no table is read."""
    return dict(db.execute("SELECT name, n FROM counts"))


def read_codes(db: sqlite3.Connection, after: int | None = None, most: int | None = None) -> "tuple[Ids, Bits] | None":
    """The codes of a store's vectors, by chunk id: (ids, rows of uint8), batch by batch; None if there are none. All of
    them, at most `most`, by default as many as the vectors the store counts, which it keeps, into room made for that
    many at once: counting the codes was a pass over millions before the one that reads them, and codes added since
    the count wait for the next read. Or those of the chunks `after` a chunk id, at most `most`, in room for those
    found: a server reads them on every search while it reads all anew."""
    import numpy as np  # here, not at the top: the CLI starts with the standard library alone

    if after is not None:
        sql = "SELECT chunk, bits FROM codes WHERE chunk > ? ORDER BY chunk LIMIT ?"
        cursor = db.execute(sql, (after, -1 if most is None else most))
        parts: list[tuple[Ids, Bits]] = []
        while batch := cursor.fetchmany(CODES_BATCH):
            ids = np.fromiter((chunk for chunk, _ in batch), np.int64, len(batch))
            parts.append((ids, np.frombuffer(b"".join(b for _, b in batch), np.uint8).reshape(len(batch), -1)))
        return (np.concatenate([i for i, _ in parts]), np.concatenate([b for _, b in parts])) if parts else None
    n = counts(db)["vectors"] if most is None else most
    ids, row = np.empty(n, np.int64), 0
    bits: Bits | None = None
    cursor = db.execute("SELECT chunk, bits FROM codes ORDER BY chunk")
    while row < n and (batch := cursor.fetchmany(CODES_BATCH)):
        batch = batch[: n - row]
        if bits is None:
            bits = np.empty((n, len(batch[0][1])), np.uint8)
        rows = np.frombuffer(b"".join(b for _, b in batch), np.uint8).reshape(len(batch), -1)
        ids[row : row + len(batch)], bits[row : row + len(batch)] = [chunk for chunk, _ in batch], rows
        row += len(batch)
    return (ids[:row], bits[:row]) if row else None  # none, or all deleted meanwhile


def add_codes(db: sqlite3.Connection) -> None:
    """Add the codes of the vectors that have none, computed from the vectors: nothing is embedded again. A batch per
    transaction, so that this goes on where it stopped, and another sift waits a batch at most. Vectors of another
    length than the store's get none: they can't be searched with the others."""
    import numpy as np  # here, not at the top: the CLI starts with the standard library alone

    uncoded = "vectors v WHERE NOT EXISTS (SELECT 1 FROM codes c WHERE c.chunk = v.chunk)"
    if missing := db.execute(f"SELECT count(*) FROM {uncoded}").fetchone()[0]:
        console.say(f"adding the codes of {console.count(missing, 'vector')}, once, which searches load instead ...")
    last = -1
    sql = f"SELECT v.chunk, v.vec FROM {uncoded} AND v.chunk > ? ORDER BY v.chunk LIMIT ?"
    size, batch = 2 * int(get_meta(db, "dim", 0)), True
    while batch:
        db.execute("BEGIN IMMEDIATE")  # before reading: another sift may change vectors until then
        try:
            if batch := db.execute(sql, (last, CODES_BATCH)).fetchall():
                last, rows = batch[-1][0], [row for row in batch if len(row[1]) == size]
                matrix = np.frombuffer(b"".join(vec for _, vec in rows), np.float16).reshape(len(rows), size // 2)
                _put_codes(db, [chunk for chunk, _ in rows], matrix)
            db.commit()
        except BaseException:
            db.rollback()
            raise


def put_vectors(db: sqlite3.Connection, chunks: Iterable[int], matrix: "Floats") -> None:
    """Store the vectors of chunks (fp16 rows of `matrix`) with their codes, replacing any they had: as an upsert,
    which keeps the count (see COUNTS)."""
    ids = [int(chunk) for chunk in chunks]
    rows = zip(ids, (vector.tobytes() for vector in matrix))
    db.executemany(
        "INSERT INTO vectors(chunk, vec) VALUES(?, ?) ON CONFLICT(chunk) DO UPDATE SET vec = excluded.vec", rows
    )
    _put_codes(db, ids, matrix)


def _put_codes(db: sqlite3.Connection, chunks: list[int], matrix: "Floats") -> None:
    """Write the codes of these chunks' vectors. Codes written at or before the last there is change those a running
    server holds: a new epoch (see codes_changed); `sift embed` writes past the last, which a server reads alone."""
    from .vectors import codes  # imports NumPy

    last = db.execute("SELECT max(chunk) FROM codes").fetchone()[0]
    if chunks and last is not None and min(chunks) <= last:
        codes_changed(db)
    bits = codes(matrix)
    db.executemany("INSERT OR REPLACE INTO codes(chunk, bits) VALUES(?, ?)", zip(chunks, (b.tobytes() for b in bits)))


def drop_vectors(db: sqlite3.Connection, where: str, args: Sequence[Any] = ()) -> int:
    """Delete the vectors, and with them their codes, of the chunks `where` (SQL on the vectors' `chunk`, with `args`)
    selects; a new epoch of the codes if any went. Every deletion of vectors goes through here. Returns how many."""
    dropped = db.execute(f"DELETE FROM vectors WHERE {where}", args).rowcount
    if dropped:
        codes_changed(db)
    return dropped


def vectors_state(db: sqlite3.Connection) -> tuple[str | None, str | None, int]:
    """What a store says of its vectors, as a server follows them: (its version, the epoch of its codes, how many
    vectors it counts)."""
    return get_meta(db, "version"), get_meta(db, "codes_epoch"), counts(db)["vectors"]


def codes_changed(db: sqlite3.Connection) -> None:
    """Mark that the codes changed otherwise than by codes written past the last: a server holding them reads them all
    again, beside its searches, where it reads only those past the last it holds as long as the epoch stays."""
    set_meta(db, "codes_epoch", uuid.uuid4().hex)


@contextmanager
def snapshot(dbs: Iterable[sqlite3.Connection]) -> Generator[None]:
    """While this lasts, each of these stores is read in one transaction: as it was at the first read, whatever other
    sifts write meanwhile (WAL). A store in a transaction already stays in it."""
    began = [db for db in dbs if not db.in_transaction]
    for db in began:
        db.execute("BEGIN")
    try:
        yield
    finally:
        for db in began:
            db.commit()  # nothing written: it ends the reading


@overload
def get_meta(db: sqlite3.Connection, key: str) -> str | None: ...


@overload
def get_meta(db: sqlite3.Connection, key: str, default: _D) -> str | _D: ...


def get_meta(db: sqlite3.Connection, key: str, default: object = None) -> object:
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(db: sqlite3.Connection, key: str, value: object) -> None:
    db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, str(value)))


def empty_log(db: sqlite3.Connection) -> None:
    """Write the write-ahead log into the store and empty it, after a change that grew it while readers held it (a
    postings build): it would keep its size while any other sift has the store open. Waits for those readers, at most
    TIMEOUT; still held then, it shrinks at a later change (WAL_LIMIT)."""
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def touch(db: sqlite3.Connection) -> None:
    """Mark the vectors as changed, so a running server follows their codes (see codes_changed). Random, as clocks can
    tick too slowly."""
    set_meta(db, "version", uuid.uuid4().hex)


def texts(db: sqlite3.Connection, spans: Iterable[tuple[int, int, int, int]]) -> dict[int, str]:
    """The texts of chunks, each given as (id, document, position, length): {id: text}, without those whose document
    is gone. As the chunk_text view cuts them, but from a text read once per document: SQLite's substr loads the whole
    text and walks to the position for every chunk it cuts, 6 ms for each of the 3,310 of an 8 MB file."""
    return dict(_cut(db, spans))


def _cut(
    db: sqlite3.Connection, spans: Iterable[tuple[int, int, int, int]], columns: str = ""
) -> Iterator[tuple[Any, ...]]:
    """(id, text, *`columns` of its document `d`, as SQL) of each chunk of `spans`, document by document."""
    chunks: dict[int, list[tuple[int, int, int]]] = {}
    for chunk, doc, pos, length in spans:
        chunks.setdefault(doc, []).append((chunk, pos, length))
    sql = f"SELECT d.id, d.text{columns} FROM docs d WHERE d.id IN (SELECT value FROM json_each(?))"
    for doc, text, *rest in db.execute(sql, (json.dumps(list(chunks)),)):
        end = text.find("\x00")  # where substr stops: sift 1.11 and older stored a text with its NULs
        text = text if end < 0 else text[:end]
        for chunk, pos, length in chunks[doc]:
            yield (chunk, text[pos : pos + length], *rest)


def _chunks(chunks: Sequence[int] | None, docs: Sequence[int] | None) -> tuple[str, tuple[str, ...]]:
    """Chunks `c` as a FROM clause, with its arguments: these `chunks` (ids), else the chunks of these `docs` (ids),
    else every chunk."""
    if chunks is not None:
        return "chunks c JOIN json_each(?) j ON c.id = j.value", (json.dumps(chunks),)
    if docs is not None:
        return "chunks c JOIN json_each(?) j ON c.doc = j.value", (json.dumps(docs),)
    return "chunks c", ()


def chunk_rows(
    db: sqlite3.Connection, chunks: Sequence[int] | None = None, docs: Sequence[int] | None = None, where: str = "1"
) -> Iterator[tuple[Any, ...]]:
    """What the chunk_text view reads of chunks (see _chunks, and of those the ones `where` holds for), each as (id,
    text, title, meta), by id. Cut as `texts` cuts them, CUT at a time as they are asked for: the rows of millions of
    chunks pass through, none held."""
    source, args = _chunks(chunks, docs)
    sql = f"SELECT id, doc, pos, len FROM chunks WHERE id IN (SELECT c.id FROM {source} WHERE {where}) ORDER BY id"
    spans = db.execute(sql, args)
    while batch := spans.fetchmany(CUT):
        yield from sorted(_cut(db, batch, f", d.title, {META}"))  # by id, as FTS5 writes them in one go


def fill(
    db: sqlite3.Connection,
    table: str,
    chunks: Sequence[int] | None = None,
    docs: Sequence[int] | None = None,
    delete: bool = False,
    count: int | None = None,
) -> None:
    """Write what the chunk_text view reads of chunks (see _chunks) into the FTS5 table `table`, or with `delete` take
    their entries out of it. The chunks of short documents in one statement, through the view, as one per document
    took twice as long; those of documents of more than LONG chunks cut by chunk_rows. `count` is how many chunks
    they are, where the caller knows: if the one statement wrote as many, no long document is looked for, a
    statement saved for each of a collection's small files."""
    source, args = _chunks(chunks, docs)
    columns, first = (f"{table}, rowid", "'delete', ") if delete else ("rowid", "")
    long = f"EXISTS (SELECT 1 FROM chunks k WHERE k.doc = c.doc LIMIT 1 OFFSET {LONG})"
    wrote = db.execute(
        f"INSERT INTO {table}({columns}, title, text, meta) SELECT {first}id, title, text, meta FROM chunk_text "
        f"WHERE id IN (SELECT c.id FROM {source} WHERE NOT {long})",
        args,
    ).rowcount
    if wrote != count:
        rows = chunk_rows(db, chunks, docs, where=long)
        db.executemany(f"INSERT INTO {table}({columns}, text, title, meta) VALUES({first}?, ?, ?, ?)", rows)


def keywords(db: sqlite3.Connection, docs: Sequence[int], add: bool = True, count: int | None = None) -> None:
    """Add the keyword index's entries for the chunks of these documents, by id, or delete them, as the chunk_text
    view reads them now; `count` is how many chunks they have, if known (see fill). Delete them before a document's
    text, title or meta changes, and add them after: entries that no longer match the view can't be deleted any
    more. Once the postings are built, the chunks are marked dirty too, until the postings follow."""
    if not docs:
        return
    fill(db, "fts", docs=docs, delete=not add, count=count)
    if get_meta(db, "kw_built") is not None:
        _mark_dirty(db, docs)


def _mark_dirty(db: sqlite3.Connection, docs: Sequence[int]) -> None:
    """Mark the chunks of these documents dirty, numbered by this change: their postings no longer follow FTS5."""
    change = int(get_meta(db, "kw_change", 0)) + 1
    set_meta(db, "kw_change", change)
    db.execute(
        "INSERT INTO kw_dirty(chunk, change) SELECT c.id, ? FROM chunks c JOIN json_each(?) j ON c.doc = j.value "
        "WHERE true ON CONFLICT(chunk) DO UPDATE SET change = excluded.change",
        (change, json.dumps(docs)),
    )


# The documents left out as another file's held their paths (a record or an attachment is a document below its file's
# name, `data.jsonl`'s record `a.md` is `data/a.md`, as a real file may be): the path, and the file of the document
# waiting, freed once the path is. Made by the first such document; a file's rows go when it is stored anew or gone.
WAITING = (
    """CREATE TABLE IF NOT EXISTS waiting(path TEXT NOT NULL, source INTEGER NOT NULL,
        freed INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(path, source))""",
    "CREATE INDEX IF NOT EXISTS waiting_source ON waiting(source)",  # each file stored asks: else a scan of them all
)


def wait(db: sqlite3.Connection, path: str, source: int) -> None:
    """Record that a document of the file `source` waits for `path`, which another file's document holds."""
    for statement in WAITING:
        db.execute(statement)
    db.execute("INSERT OR IGNORE INTO waiting(path, source) VALUES(?, ?)", (path, source))


def stop_waiting(db: sqlite3.Connection, source: int) -> None:
    """Forget what the documents of the file `source` waited for: it is being stored anew, and says so again if they
    still wait, or it is gone."""
    if _waits(db):
        db.execute("DELETE FROM waiting WHERE source = ?", (source,))


def freed(db: sqlite3.Connection) -> list[str]:
    """The paths of the files whose documents waited for a path freed since this was last asked (see _free)."""
    if not _waits(db):
        return []
    sql = "SELECT DISTINCT s.path FROM waiting w JOIN sources s ON s.id = w.source WHERE w.freed"
    if found := [path for (path,) in db.execute(sql)]:
        db.execute("UPDATE waiting SET freed = 0 WHERE freed")  # written only then: a write waits for other writers
    return found


def _waits(db: sqlite3.Connection) -> bool:
    return db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'waiting'").fetchone() is not None


def _free(db: sqlite3.Connection, paths: Sequence[str]) -> None:
    """Mark the files waiting for these paths, freed (see wait), to be read again: their times cleared, for the next
    `sift index`, and `freed` for one running."""
    if paths and _waits(db):
        marks = ",".join("?" * len(paths))
        db.execute(f"UPDATE waiting SET freed = 1 WHERE path IN ({marks})", paths)
        db.execute(
            f"UPDATE sources SET mtime = NULL WHERE id IN (SELECT source FROM waiting WHERE path IN ({marks}))", paths
        )


def delete_docs(db: sqlite3.Connection, doc_ids: Sequence[int], free: bool = True) -> None:
    """Delete documents, with all that is theirs; the files waiting for their paths are read again (see _free), unless
    not `free`, as for a document replaced by its new version at once."""
    for i in range(0, len(doc_ids), 500):
        batch = doc_ids[i : i + 500]
        marks = ",".join("?" * len(batch))
        if free:
            _free(db, [path for (path,) in db.execute(f"SELECT path FROM docs WHERE id IN ({marks})", batch)])
        chunk_ids = f"SELECT id FROM chunks WHERE doc IN ({marks})"
        keywords(db, batch, add=False)  # first, while their text is still there
        drop_vectors(db, f"chunk IN ({chunk_ids})", batch)
        db.execute(f"DELETE FROM links WHERE doc IN ({marks})", batch)
        db.execute(f"DELETE FROM names WHERE doc IN ({marks})", batch)
        db.execute(f"DELETE FROM chunks WHERE doc IN ({marks})", batch)
        db.execute(f"DELETE FROM docs WHERE id IN ({marks})", batch)
    if doc_ids:
        touch(db)
