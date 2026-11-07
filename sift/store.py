"""A collection's database, its store: its documents, chunks, keyword index (FTS5), vectors and their codes, and its
notes' links, in one SQLite file. The catalog (see `catalog`) lists the collections and where their stores lie.

A store holds the text of every document, so searching and reading never touch the files; they are read only when
indexing (see `indexing`, and `readers` for the kinds of file).

The ids of a collection's documents and chunks start at its id << SHIFT: unique across collections, they say whose
they are (`owner`), so search results from several stores merge as they are.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import console, readers

SCHEMA_VERSION = 7  # in SQLite's user_version; up to sift 1.7, one database held every collection (see `upgrade`)
FIRST_LAYOUT = 6  # sift 1.8's, the first of a store per collection; its rows are counted once (see _count_rows)
TIMEOUT = 30  # seconds to wait for another sift's transaction
UPGRADE_WAIT = 3600  # seconds to wait while another sift counts a store's rows, once: minutes for a large one
SHIFT = 40  # a collection's row ids start at its id << SHIFT: a trillion rows each, and millions of collections
# The keyword index reads chunks through a view: each with its document's title, and its front-matter values but the
# title, e.g. authors and ids, as `meta`. Inserting and deleting read it alike, so the two always agree.
KEYWORD_INDEX = """
CREATE VIEW IF NOT EXISTS chunk_text AS SELECT c.id AS id, d.title AS title, substr(d.text, c.pos + 1, c.len) AS text,
    (SELECT group_concat(j.atom, ' ') FROM json_tree(d.meta) j WHERE j.atom IS NOT NULL AND j.fullkey != '$.title')
    AS meta FROM chunks c JOIN docs d ON d.id = c.doc;
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(title, text, meta, content='chunk_text', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2');
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
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("PRAGMA cache_size=-262144")  # 256 MB
    except BaseException:
        db.close()
        raise
    return db


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
    if version < SCHEMA_VERSION:
        _count_rows(db, path)
    for key, kinds in READ_AGAIN.items():
        if get_meta(db, key) is None:
            _read_again(db, key, kinds)


def make(db, cid, read=True):
    """The tables of a new store of the collection `cid`, its ids starting at first_id(cid). Unless `read`, its files
    of the kinds an older sift skipped are still to be read again (READ_AGAIN)."""
    db.executescript(SCHEMA + KEYWORD_INDEX + LINKS + CODES + "".join(f"{statement};\n" for statement in COUNTS))
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
            if db.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
                console.say(f"counting the rows of {path} once, to keep them counted from now on ...")
                for statement in COUNTS:
                    db.execute(statement)
                rows = [(table, db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]) for table in COUNTED]
                db.executemany("INSERT INTO counts(name, n) VALUES(?, ?)", rows)
                db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            db.commit()
        except BaseException:
            db.rollback()
            raise
    finally:
        db.execute(f"PRAGMA busy_timeout = {TIMEOUT * 1000}")


def counts(db):
    """How many rows of each table in COUNTED the store holds, by table, as kept: no table is read."""
    return dict(db.execute("SELECT name, n FROM counts"))


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
    from .vectors import codes  # imports NumPy

    bits = codes(matrix)
    db.executemany("INSERT OR REPLACE INTO codes(chunk, bits) VALUES(?, ?)", zip(chunks, (b.tobytes() for b in bits)))


def get_meta(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, str(value)))


def touch(db):
    """Mark the vectors as changed, so a running server reloads their codes. Random, as clocks can tick too slowly."""
    set_meta(db, "version", uuid.uuid4().hex)


def keywords(db, docs, add=True):
    """Add the keyword index's entries for the chunks of these documents, by id, or delete them, as the chunk_text
    view reads them now. Delete them before a document's text, title or meta changes, and add them after: entries
    that no longer match the view can't be deleted any more. One statement for all, as one per document took twice
    as long."""
    columns, values = ("rowid", "id") if add else ("fts, rowid", "'delete', id")
    db.execute(
        f"INSERT INTO fts({columns}, title, text, meta) SELECT {values}, title, text, meta FROM chunk_text "
        "WHERE id IN (SELECT c.id FROM chunks c JOIN json_each(?) j ON c.doc = j.value)",
        (json.dumps(docs),),
    )


def delete_docs(db, doc_ids):
    for i in range(0, len(doc_ids), 500):
        batch = doc_ids[i : i + 500]
        marks = ",".join("?" * len(batch))
        chunk_ids = f"SELECT id FROM chunks WHERE doc IN ({marks})"
        keywords(db, batch, add=False)  # first, while their text is still there
        db.execute(f"DELETE FROM vectors WHERE chunk IN ({chunk_ids})", batch)
        db.execute(f"DELETE FROM links WHERE doc IN ({marks})", batch)
        db.execute(f"DELETE FROM names WHERE doc IN ({marks})", batch)
        db.execute(f"DELETE FROM chunks WHERE doc IN ({marks})", batch)
        db.execute(f"DELETE FROM docs WHERE id IN ({marks})", batch)
    if doc_ids:
        touch(db)
