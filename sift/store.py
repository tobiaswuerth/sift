"""A collection's database, its store: its documents, chunks, keyword index (FTS5) and keyword postings, vectors and
their codes, its notes' links, and the counts of its rows, in one SQLite file. The catalog (see `catalog`) lists the
collections and where their stores lie.

A store holds the text of every document, so searching and reading never touch the files; they are read only when
indexing (see `indexing`, and `readers` for the kinds of file).

The ids of a collection's documents and chunks start at its id << SHIFT: unique across collections, they say whose
they are (`owner`), so search results from several stores merge as they are.
"""

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import console, readers

SCHEMA_VERSION = 9  # in SQLite's user_version; up to sift 1.7, one database held every collection (see `upgrade`)
FIRST_LAYOUT = 6  # sift 1.8's, the first of a store per collection; its rows are counted once (see _count_rows)
COUNTED_LAYOUT = 7  # sift 1.9's, which counts rows; 8 adds the keyword postings' tables (POSTINGS), 9 their order
TIMEOUT = 30  # seconds to wait for another sift's transaction
WAL_LIMIT = 64 * 2**20  # bytes the write-ahead log shrinks back to once written into the store: SQLite keeps its size
UPGRADE_WAIT = 3600  # seconds to wait while another sift counts a store's rows, once: minutes for a large one
# While set, every statement on every database stops with "interrupted": a server sets it for a call its client
# cancelled (see `jobs`). SQLite looks every ABORT_STEPS instructions, about a millisecond.
ABORT = threading.Event()
ABORT_STEPS = 100_000
SHIFT = 40  # a collection's row ids start at its id << SHIFT: a trillion rows each, and millions of collections
# The keyword index reads chunks through a view: each with its document's title, and its front-matter values but the
# title, e.g. authors and ids, as `meta`. Inserting and deleting read it alike, so the two always agree.
TOKENIZER = "unicode61 remove_diacritics 2"  # FTS5's, as the keyword postings tokenize too (see `postings`)
KEYWORD_INDEX = f"""
CREATE VIEW IF NOT EXISTS chunk_text AS SELECT c.id AS id, d.title AS title, substr(d.text, c.pos + 1, c.len) AS text,
    (SELECT group_concat(j.atom, ' ') FROM json_tree(d.meta) j WHERE j.atom IS NOT NULL AND j.fullkey != '$.title')
    AS meta FROM chunks c JOIN docs d ON d.id = c.doc;
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(title, text, meta, content='chunk_text', content_rowid='id',
    tokenize='{TOKENIZER}');
"""
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


def postings_lock(path):
    """The file beside the store at `path` whose lock is held while its keyword postings change (postings._alone)."""
    return Path(f"{path}.postings-lock")


def files(path):
    """The files of the store at `path`: the database, its write-ahead log and shared memory, and its postings' lock."""
    return [Path(path), Path(f"{path}-wal"), Path(f"{path}-shm"), postings_lock(path)]


def first_id(cid):
    """The first row id of the collection `cid`."""
    return cid << SHIFT


def owner(row_id):
    """The id of the collection a document or chunk, by its row id, belongs to."""
    return int(row_id) >> SHIFT


def database(path, create=False) -> sqlite3.Connection:
    """The SQLite database at `path`, with sift's settings. It is made only if `create`: a store on a drive that isn't
    connected stays missing, and is never made anew, empty."""
    path = Path(path)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=TIMEOUT, check_same_thread=False)
    elif not path.is_file():
        raise FileNotFoundError(f"not found: {path}")
    else:
        db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=rw", uri=True, timeout=TIMEOUT, check_same_thread=False)
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


def read_only(path) -> sqlite3.Connection:
    """The database at `path`, to read only, through a connection of its own: no setting changes, and no call
    cancelled meanwhile (ABORT) stops what it reads."""
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, timeout=TIMEOUT)


def connect(path, cid, create=False) -> sqlite3.Connection:
    """The store of the collection `cid` at `path`; with `create`, a new one, made there."""
    db = database(path, create)
    try:
        _layout(db, path, cid, create)
    except BaseException:
        db.close()
        raise
    return db


def _layout(db, path, cid, create):
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
        _add_postings_tables(db)
    for key, kinds in READ_AGAIN.items():
        if get_meta(db, key) is None:
            _read_again(db, key, kinds)


def make(db, cid, read=True):
    """The tables of a new store of the collection `cid`, its ids starting at first_id(cid). Unless `read`, its files
    of the kinds an older sift skipped are still to be read again (READ_AGAIN)."""
    db.executescript(
        SCHEMA + KEYWORD_INDEX + LINKS + CODES + POSTINGS + "".join(f"{statement};\n" for statement in COUNTS)
    )
    db.executemany("INSERT INTO counts(name, n) VALUES(?, 0)", [(table,) for table in COUNTED])
    first = first_id(cid)
    db.executemany("INSERT INTO sqlite_sequence(name, seq) VALUES(?, ?)", [("docs", first), ("chunks", first)])
    set_meta(db, "collection", cid)
    for key in READ_AGAIN if read else ():
        set_meta(db, key, 1)
    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


@contextmanager
def alone(path, name):
    """The store at `path` of the collection `name`, held so that no other sift reads or writes it until this ends;
    its write-ahead log is written into it first, so that the file holds it all. An error if another sift has it
    open."""
    db = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=rw", uri=True, timeout=1, isolation_level=None)
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


READ_AGAIN = {  # in the meta table, once the files of these kinds that an older sift skipped are to be read again
    "scans_read": (".pdf",),  # sift 1.2 and older read no scans
    "documents_read": readers.DOCUMENTS,  # the server of sift 1.3.0 and older skipped every document it reindexed
}


def _read_again(db, key, kinds):
    """Have the next `sift index` read again the files of these kinds without documents, and the mails with one
    attached; once, as the meta `key` records. Their unchanged documents stay, vectors included. In one transaction,
    which another sift starting at the same time waits for, and then finds done."""
    db.execute("BEGIN IMMEDIATE")
    try:
        if get_meta(db, key) is None:
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
            set_meta(db, key, 1)
        db.commit()
    except BaseException:
        db.rollback()
        raise


def _count_rows(db, path):
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


def _add_postings_tables(db):
    """Add the keyword postings' tables a store lacks: all of them to a store of sift 1.9 (layout 7), unbuilt, so that
    keyword search reads FTS5 until `sift index` builds them (see `postings`); their order to one of 1.10 (layout 8),
    empty, which keeps its postings' chunks in the order they were indexed."""
    db.execute("BEGIN IMMEDIATE")
    try:
        if db.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
            for statement in filter(str.strip, POSTINGS.split(";")):  # not executescript, which would commit first
                db.execute(statement)
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        db.commit()
    except BaseException:
        db.rollback()
        raise


def counts(db):
    """How many rows of each table in COUNTED the store holds, by table, as kept: no table is read."""
    return dict(db.execute("SELECT name, n FROM counts"))


def read_codes(db, after=None, most=None):
    """The codes of a store's vectors, by chunk id: (ids, rows of uint8), batch by batch; None if there are none. All of
    them, at most `most`, by default as many as the vectors the store counts, which it keeps, into room made for that
    many at once: counting the codes was a pass over millions before the one that reads them, and codes added since
    the count wait for the next read. Or those of the chunks `after` a chunk id, at most `most`, in room for those
    found: a server reads them on every search while it reads all anew."""
    import numpy as np  # here, not at the top: the CLI starts with the standard library alone

    if after is not None:
        sql = "SELECT chunk, bits FROM codes WHERE chunk > ? ORDER BY chunk LIMIT ?"
        cursor, parts = db.execute(sql, (after, -1 if most is None else most)), []
        while batch := cursor.fetchmany(CODES_BATCH):
            ids = np.fromiter((chunk for chunk, _ in batch), np.int64, len(batch))
            parts.append((ids, np.frombuffer(b"".join(b for _, b in batch), np.uint8).reshape(len(batch), -1)))
        return (np.concatenate([i for i, _ in parts]), np.concatenate([b for _, b in parts])) if parts else None
    n = counts(db)["vectors"] if most is None else most
    ids, bits, row = np.empty(n, np.int64), None, 0
    cursor = db.execute("SELECT chunk, bits FROM codes ORDER BY chunk")
    while row < n and (batch := cursor.fetchmany(CODES_BATCH)):
        batch = batch[: n - row]
        if bits is None:
            bits = np.empty((n, len(batch[0][1])), np.uint8)
        rows = np.frombuffer(b"".join(b for _, b in batch), np.uint8).reshape(len(batch), -1)
        ids[row : row + len(batch)], bits[row : row + len(batch)] = [chunk for chunk, _ in batch], rows
        row += len(batch)
    return (ids[:row], bits[:row]) if row else None  # none, or all deleted meanwhile


def add_codes(db):
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


def put_vectors(db, chunks, matrix):
    """Store the vectors of chunks (fp16 rows of `matrix`) with their codes, replacing any they had: as an upsert,
    which keeps the count (see COUNTS)."""
    chunks = [int(chunk) for chunk in chunks]
    rows = zip(chunks, (vector.tobytes() for vector in matrix))
    db.executemany(
        "INSERT INTO vectors(chunk, vec) VALUES(?, ?) ON CONFLICT(chunk) DO UPDATE SET vec = excluded.vec", rows
    )
    _put_codes(db, chunks, matrix)


def _put_codes(db, chunks, matrix):
    """Write the codes of these chunks' vectors. Codes written at or before the last there is change those a running
    server holds: a new epoch (see codes_changed); `sift embed` writes past the last, which a server reads alone."""
    from .vectors import codes  # imports NumPy

    last = db.execute("SELECT max(chunk) FROM codes").fetchone()[0]
    if chunks and last is not None and min(chunks) <= last:
        codes_changed(db)
    bits = codes(matrix)
    db.executemany("INSERT OR REPLACE INTO codes(chunk, bits) VALUES(?, ?)", zip(chunks, (b.tobytes() for b in bits)))


def drop_vectors(db, where, args=()):
    """Delete the vectors, and with them their codes, of the chunks `where` (SQL on the vectors' `chunk`, with `args`)
    selects; a new epoch of the codes if any went. Every deletion of vectors goes through here. Returns how many."""
    dropped = db.execute(f"DELETE FROM vectors WHERE {where}", args).rowcount
    if dropped:
        codes_changed(db)
    return dropped


def vectors_state(db):
    """What a store says of its vectors, as a server follows them: (its version, the epoch of its codes, how many
    vectors it counts)."""
    return get_meta(db, "version"), get_meta(db, "codes_epoch"), counts(db)["vectors"]


def codes_changed(db):
    """Mark that the codes changed otherwise than by codes written past the last: a server holding them reads them all
    again, beside its searches, where it reads only those past the last it holds as long as the epoch stays."""
    set_meta(db, "codes_epoch", uuid.uuid4().hex)


def get_meta(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, str(value)))


def empty_log(db):
    """Write the write-ahead log into the store and empty it, after a change that grew it while readers held it (a
    postings build): it would keep its size while any other sift has the store open. Waits for those readers, at most
    TIMEOUT; still held then, it shrinks at a later change (WAL_LIMIT)."""
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def touch(db):
    """Mark the vectors as changed, so a running server follows their codes (see codes_changed). Random, as clocks can
    tick too slowly."""
    set_meta(db, "version", uuid.uuid4().hex)


def keywords(db, docs, add=True):
    """Add the keyword index's entries for the chunks of these documents, by id, or delete them, as the chunk_text
    view reads them now. Delete them before a document's text, title or meta changes, and add them after: entries
    that no longer match the view can't be deleted any more. One statement for all, as one per document took twice
    as long. Once the postings are built, the chunks are marked dirty too, until the postings follow."""
    columns, values = ("rowid", "id") if add else ("fts, rowid", "'delete', id")
    db.execute(
        f"INSERT INTO fts({columns}, title, text, meta) SELECT {values}, title, text, meta FROM chunk_text "
        "WHERE id IN (SELECT c.id FROM chunks c JOIN json_each(?) j ON c.doc = j.value)",
        (json.dumps(docs),),
    )
    if docs and get_meta(db, "kw_built") is not None:
        _mark_dirty(db, docs)


def _mark_dirty(db, docs):
    """Mark the chunks of these documents dirty, numbered by this change: their postings no longer follow FTS5."""
    change = int(get_meta(db, "kw_change", 0)) + 1
    set_meta(db, "kw_change", change)
    db.execute(
        "INSERT INTO kw_dirty(chunk, change) SELECT c.id, ? FROM chunks c JOIN json_each(?) j ON c.doc = j.value "
        "WHERE true ON CONFLICT(chunk) DO UPDATE SET change = excluded.change",
        (change, json.dumps(docs)),
    )


def delete_docs(db, doc_ids):
    for i in range(0, len(doc_ids), 500):
        batch = doc_ids[i : i + 500]
        marks = ",".join("?" * len(batch))
        chunk_ids = f"SELECT id FROM chunks WHERE doc IN ({marks})"
        keywords(db, batch, add=False)  # first, while their text is still there
        drop_vectors(db, f"chunk IN ({chunk_ids})", batch)
        db.execute(f"DELETE FROM links WHERE doc IN ({marks})", batch)
        db.execute(f"DELETE FROM names WHERE doc IN ({marks})", batch)
        db.execute(f"DELETE FROM chunks WHERE doc IN ({marks})", batch)
        db.execute(f"DELETE FROM docs WHERE id IN ({marks})", batch)
    if doc_ids:
        touch(db)
