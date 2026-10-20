"""The SQLite index: one database holding every collection, document, chunk, the keyword index (FTS5) and the vectors.

The index holds the text of every document, so searching and reading never touch the files; they are read only when
indexing (see `indexing`, and `readers` for the kinds of file).
"""

import json
import re
import sqlite3
import uuid
from pathlib import Path

from . import settings

SCHEMA_VERSION = 2  # in SQLite's user_version; sift 0.2 and older had none
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
-- a chunk is a span of its document's text, which is stored once; the keyword index reads it through this view
CREATE VIEW IF NOT EXISTS chunk_text AS SELECT c.id AS id, d.title AS title, substr(d.text, c.pos + 1, c.len) AS text
    FROM chunks c JOIN docs d ON d.id = c.doc;
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(title, text, content='chunk_text', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2');
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
    if version < SCHEMA_VERSION and db.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]:
        raise SystemExit(
            f"this index was made by an older sift, which stored documents differently: delete {path}, "
            "then run `sift add`, `sift index` and `sift embed` again"
        )
    if version < SCHEMA_VERSION:
        db.executescript(SCHEMA + f"PRAGMA user_version = {SCHEMA_VERSION};")


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
    """A collection's choice of JSON fields, {"id": name, "title": name, "text": [names]}, as stored."""
    if not fields:
        return None
    for key, value in fields.items():
        if key not in ("id", "title", "text"):
            raise SystemExit(f"unknown field choice {key!r}: choose the id, title or text field")
        names = value if key == "text" else [value]
        if (
            not (isinstance(value, list) or key != "text")
            or not names
            or not all(isinstance(n, str) and n for n in names)
        ):
            raise SystemExit(
                f"the {key} field must be {'one or more field names' if key == 'text' else 'a field name'}, "
                f"got {value!r}"
            )
    return json.dumps(fields)


def add_collection(db, path, name=None, fields=None):
    """Register a folder. `fields` chooses JSON Lines fields instead of the defaults: {"id", "title", "text"}."""
    stored = _fields(fields)
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"not a directory: {path}")
    name = name or path.name
    if not re.fullmatch(r"[\w.-]+", name):
        raise SystemExit(f"invalid collection name {name!r}: choose one with --name, of letters, digits, '_', '-', '.'")
    try:
        db.execute("INSERT INTO collections(name, path, fields) VALUES(?, ?, ?)", (name, str(path), stored))
    except sqlite3.IntegrityError:
        raise SystemExit(f"collection {name!r} already exists") from None
    db.commit()
    return name


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


def delete_docs(db, doc_ids):
    for i in range(0, len(doc_ids), 500):
        batch = doc_ids[i : i + 500]
        marks = ",".join("?" * len(batch))
        chunk_ids = f"SELECT id FROM chunks WHERE doc IN ({marks})"
        # the keyword index reads text from chunk_text, so its entries go first, while their text is still there
        db.execute(
            f"INSERT INTO fts(fts, rowid, title, text) SELECT 'delete', id, title, text FROM chunk_text "
            f"WHERE id IN ({chunk_ids})",
            batch,
        )
        db.execute(f"DELETE FROM vectors WHERE chunk IN ({chunk_ids})", batch)
        db.execute(f"DELETE FROM chunks WHERE doc IN ({marks})", batch)
        db.execute(f"DELETE FROM docs WHERE id IN ({marks})", batch)
    if doc_ids:
        touch(db)
