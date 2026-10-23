"""The SQLite index: one database holding every collection, document, chunk, the keyword index (FTS5) and the vectors.

The index holds the text of every document, so searching and reading never touch the files; they are read only when
indexing (see `indexing`, and `readers` for the kinds of file).
"""

import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path

from . import console, readers, settings

SCHEMA_VERSION = 4  # in SQLite's user_version; sift 0.2 and older had none, 0.3 to 0.7 had 2, 0.8 to 1.0 had 3
# The keyword index reads chunks through a view: each with its document's title, and its front-matter values but the
# title, e.g. authors and ids, as `meta`. Inserting and deleting read it alike, so the two always agree.
KEYWORD_INDEX = """
CREATE VIEW IF NOT EXISTS chunk_text AS SELECT c.id AS id, d.title AS title, substr(d.text, c.pos + 1, c.len) AS text,
    (SELECT group_concat(j.atom, ' ') FROM json_tree(d.meta) j WHERE j.atom IS NOT NULL AND j.fullkey != '$.title')
    AS meta FROM chunks c JOIN docs d ON d.id = c.doc;
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(title, text, meta, content='chunk_text', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2');
"""
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS collections(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, path TEXT NOT NULL,
    fields TEXT);
CREATE TABLE IF NOT EXISTS sources(id INTEGER PRIMARY KEY, collection INTEGER NOT NULL, path TEXT NOT NULL,
    mtime INTEGER, size INTEGER, UNIQUE(collection, path));
CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, collection INTEGER NOT NULL, source INTEGER NOT NULL,
    path TEXT NOT NULL, hash TEXT, title TEXT, meta TEXT, text TEXT NOT NULL, UNIQUE(collection, path));
CREATE INDEX IF NOT EXISTS docs_source ON docs(source);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, doc INTEGER NOT NULL, seq INTEGER,
    line_start INTEGER, line_end INTEGER, pos INTEGER NOT NULL, len INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc);
CREATE TABLE IF NOT EXISTS vectors(chunk INTEGER PRIMARY KEY, vec BLOB NOT NULL);
"""
# The links of Markdown notes as written, and the names files answer to, both by key: see `links`, which resolves the
# one to the other when asked.
LINKS = """
CREATE TABLE IF NOT EXISTS links(id INTEGER PRIMARY KEY, doc INTEGER NOT NULL, collection INTEGER NOT NULL,
    field TEXT, line INTEGER NOT NULL, kind TEXT NOT NULL, target TEXT NOT NULL, key TEXT NOT NULL, fragment TEXT,
    shown TEXT);
CREATE INDEX IF NOT EXISTS links_doc ON links(doc);
CREATE INDEX IF NOT EXISTS links_key ON links(collection, key);
CREATE TABLE IF NOT EXISTS names(doc INTEGER NOT NULL, collection INTEGER NOT NULL, key TEXT NOT NULL,
    kind TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS names_doc ON names(doc);
CREATE INDEX IF NOT EXISTS names_key ON names(collection, key);
"""


def connect(home=None) -> sqlite3.Connection:
    home = Path(home or settings.current().home)
    home.mkdir(parents=True, exist_ok=True)
    path = home / "sift.db"
    db = sqlite3.connect(path, timeout=30, check_same_thread=False)
    try:
        _layout(db, path)
    except BaseException:
        db.close()
        raise
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-262144")  # 256 MB
    return db


def _layout(db, path):
    """Create the tables of a new index, or check that an existing one has this version's layout."""
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise SystemExit(f"{path} was made by a newer sift; upgrade sift to use it")
    if version < 2 and db.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]:
        raise SystemExit(
            f"this index was made by an older sift, which stored documents differently: delete {path}, "
            "then run `sift add`, `sift index` and `sift embed` again"
        )
    if version < 2:
        db.executescript(SCHEMA + KEYWORD_INDEX + LINKS + f"PRAGMA user_version = {SCHEMA_VERSION};")
        for key in READ_AGAIN:
            set_meta(db, key, 1)
        db.commit()
        return
    if version == 2:  # sift 0.3 to 0.7: everything stays but the keyword index, which gains the front matter
        _rebuild_keyword_index(db)
    if version <= 3:  # sift 0.8 to 1.0: everything stays, and the notes' links join it
        _add_links(db)
    for key, kinds in READ_AGAIN.items():
        if get_meta(db, key) is None:
            _read_again(db, key, kinds)


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


def _rebuild_keyword_index(db):
    """Make the keyword index anew from the documents, in one transaction: a few minutes for millions of chunks."""
    console.say("updating the keyword index to this version of sift, a few minutes for a large index ...")
    start = time.perf_counter()
    db.executescript(  # FTS5's own 'rebuild' fails on the view: "no such table: main.json_tree"
        "BEGIN; DROP TABLE fts; DROP VIEW chunk_text;"
        + KEYWORD_INDEX
        + "INSERT INTO fts(rowid, title, text, meta) SELECT id, title, text, meta FROM chunk_text;"
        + "PRAGMA user_version = 3; COMMIT;"
    )
    console.say(f"keyword index updated in {console.duration(time.perf_counter() - start)}")


def _add_links(db):
    """Add the links of Markdown notes and the names files answer to, read from the text the index holds: no file is
    read, nothing embedded again (see indexing.add_links). In one transaction, which another sift starting at the same
    time waits for, and then finds done."""
    from . import indexing  # it reads notes as indexing does

    db.execute("BEGIN IMMEDIATE")
    try:
        if db.execute("PRAGMA user_version").fetchone()[0] == 3:
            for statement in ["DROP TABLE IF EXISTS links", "DROP TABLE IF EXISTS names", *LINKS.split(";")[:-1]]:
                db.execute(statement)
            indexing.add_links(db)
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        db.commit()
    except BaseException:
        db.rollback()
        raise


def get_meta(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, str(value)))


def touch(db):
    """Mark the vectors as changed, so a running server reloads them. Random, as clocks can tick too slowly."""
    set_meta(db, "version", uuid.uuid4().hex)


# --- collections -------------------------------------------------------------


def _fields(fields):
    """A collection's choices, as stored: JSON Lines fields, {"id": name, "title": name, "text": [names]}, and where
    its Markdown and text files take their titles from, {"title_from": "heading" or "name"}."""
    if not fields:
        return None
    for key, value in fields.items():
        if key == "title_from":
            if value not in readers.TITLE_FROM:
                raise SystemExit(f"titles come from 'heading' or 'name', got {value!r}")
            continue
        if key not in ("id", "title", "text"):
            raise SystemExit(f"unknown field choice {key!r}: choose the id, title or text field")
        names = value if key == "text" else [value]
        if not (isinstance(names, list) and names and all(isinstance(n, str) and n for n in names)):
            raise SystemExit(
                f"the {key} field must be {'one or more field names' if key == 'text' else 'a field name'}, "
                f"got {value!r}"
            )
    return json.dumps(fields)


def add_collection(db, path, name=None, fields=None, nested=False):
    """Register a folder. `fields` chooses JSON Lines fields instead of the defaults: {"id", "title", "text"}. A folder
    inside a collection, or one holding a collection, only if `nested`: the files of both would be indexed twice."""
    stored = _fields(fields)
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"not a directory: {path}")
    if other := collection_at(db, path):  # registered twice, its documents were found twice
        raise SystemExit(f"{path} is the collection {other!r} already")
    if not nested and (overlap := _overlapping(db, path)):
        other, folder, inside = overlap
        where = "is inside" if inside else "holds"
        raise SystemExit(
            f"{path} {where} the collection {other!r} ({folder}): its files would be indexed twice, once in each "
            "collection, and found twice; add it anyway with --nested"
        )
    name = name or path.name
    if not re.fullmatch(r"[\w.-]+", name):
        raise SystemExit(f"invalid collection name {name!r}: choose one with --name, of letters, digits, '_', '-', '.'")
    try:
        db.execute("INSERT INTO collections(name, path, fields) VALUES(?, ?, ?)", (name, str(path), stored))
    except sqlite3.IntegrityError:
        raise SystemExit(f"collection {name!r} already exists") from None
    db.commit()
    return name


def no_collection(db, name):
    """The error for a collection the index doesn't have: it names those it has."""
    names = [n for (n,) in db.execute("SELECT name FROM collections ORDER BY name")]
    there = f"the collections are {', '.join(names)}" if names else "none is registered yet: `sift add FOLDER`"
    return LookupError(f"no collection named {name!r}; {there}")


def collection_at(db, path):
    """The name of the collection of the folder `path`, or None."""
    path = os.path.normcase(Path(path).expanduser().resolve())
    return next(
        (name for name, folder in db.execute("SELECT name, path FROM collections") if os.path.normcase(folder) == path),
        None,
    )


def _overlapping(db, path):
    """A collection whose folder holds the folder `path`, or lies inside it: (its name, its folder, whether `path` is
    inside it); None if there is none."""
    mine = Path(os.path.normcase(path))
    for name, folder in db.execute("SELECT name, path FROM collections ORDER BY name"):
        theirs = Path(os.path.normcase(folder))
        if mine.is_relative_to(theirs) or theirs.is_relative_to(mine):
            return name, folder, mine.is_relative_to(theirs)
    return None


def set_title_from(db, name, title_from):
    """Choose where a collection's Markdown and text files take their titles from. Returns the collection's id and
    its choices."""
    row = db.execute("SELECT id, fields FROM collections WHERE name = ?", (name,)).fetchone()
    if not row:
        raise SystemExit(f"no collection named {name!r}")
    fields = json.loads(row[1]) if row[1] else {}
    fields = {key: value for key, value in fields.items() if key != "title_from"} | {"title_from": title_from}
    db.execute("UPDATE collections SET fields = ? WHERE id = ?", (_fields(fields), row[0]))
    return row[0], fields


def remove_collection(db, name):
    """Remove a collection from the index; its files stay on disk. Returns the number of documents removed."""
    row = db.execute("SELECT id FROM collections WHERE name=?", (name,)).fetchone()
    if not row:
        raise SystemExit(f"no collection named {name!r}")
    docs = [d for (d,) in db.execute("SELECT id FROM docs WHERE collection=?", row)]
    delete_docs(db, docs)
    db.execute("DELETE FROM sources WHERE collection=?", row)
    db.execute("DELETE FROM collections WHERE id=?", row)
    db.commit()
    return len(docs)


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
