"""SQLite store: collections, documents, chunks, keyword index and vectors.

The index holds the text of every document, so searching and reading never touch the files; they are read only
when indexing. Folders hold Markdown and text files, one document each, and JSON Lines files, one document per line.
"""

import hashlib
import json
import multiprocessing
import os
import re
import sqlite3
import sys
import time
import uuid
from collections import Counter, namedtuple
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from itertools import repeat
from pathlib import Path

HOME = Path(os.environ.get("SIFT_HOME", Path.home() / ".sift"))
JSONL = (".jsonl", ".ndjson")
EXTS = (".md", ".markdown", ".txt", *JSONL)
SKIP_DIRS = {"node_modules", "__pycache__"}
CHUNK_CHARS, OVERLAP_CHARS = 3000, 300  # ~750 tokens, well inside the models' 1024-token window
WRITE_BATCH = 5000  # files per commit
BATCH_BYTES = 256 << 20  # and at most this much file content, for big JSON Lines files
FILES_PER_WORKER = 1000  # a worker process only pays off for at least this many files
BYTES_PER_WORKER = 32 << 20  # or this much content

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
    home = Path(home or HOME)
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


# --- parsing and chunking ----------------------------------------------------

FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(?:(.*?)\r?\n)??---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
LIST_ITEM = re.compile(r"[ \t]*-[ \t]+(.*)")
HEADING = re.compile(r" {0,3}#{1,6}[ \t]")  # how a Markdown (ATX) heading starts
FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")  # how a fenced code block starts and ends
LINE = re.compile(r"[^\n]*\n|[^\n]+")  # only \n ends a line, as in editors; str.splitlines knows eight more


def _value(text):
    """A front-matter value: JSON if it parses (lists, numbers, quoted strings), else plain text."""
    try:
        return json.loads(text, parse_constant=str)  # NaN/Infinity stay strings: SQLite's JSON rejects them
    except ValueError:
        return text.strip("'\"")


def _front_matter(block):
    """Flat `key: value` lines, and lists as `- item` lines below a key without a value, as Obsidian writes them."""
    meta, listing = {}, None
    for line in block.splitlines():
        if listing and (item := LIST_ITEM.match(line)):
            meta[listing] = [*(meta[listing] or []), _value(item.group(1).strip())]
            continue
        key, sep, value = line.partition(":")
        if sep and key.strip() and not key[0].isspace() and key[0] != "#":  # not nested, not a comment
            key, value = key.strip(), value.strip()
            meta[key] = _value(value) if value else None
            listing = None if value else key
    return meta


def _heading(text):
    """The first Markdown heading outside fenced code blocks, or None."""
    fence = None
    for line in text.split("\n"):
        if m := FENCE.match(line):
            mark = m.group(1)
            if not fence:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence) and not line[m.end() :].strip():
                fence = None
        elif not fence and HEADING.match(line):
            title = line.strip().lstrip("#").strip()
            if (bare := title.rstrip("#")) != title and (not bare or bare[-1] in " \t"):
                title = bare.strip()  # a closing sequence: "## Title ##", but not "# C#"
            if title:
                return title
    return None


def parse(text, fallback_title):
    """Split off YAML front matter. Returns title, meta, body and the body's first line number."""
    meta, first_line = {}, 1
    m = FRONT_MATTER.match(text)
    if m:
        meta = _front_matter(m.group(1) or "")
        first_line += text.count("\n", 0, m.end())
        text = text[m.end() :]
    title = meta.get("title") if isinstance(meta.get("title"), str) and meta["title"].strip() else None
    return title or _heading(text) or fallback_title, meta, text, first_line


def chunk(text, first_line=1):
    """Split text into pieces of at most CHUNK_CHARS on line boundaries, preferring headings, with a small overlap.

    Yields (first line, last line, position, length): each piece is text[position:position + length], stripped.
    """
    lines, pos = [], 0  # (line number, position, line)
    for n, line in enumerate(LINE.findall(text), first_line):
        while len(line) > CHUNK_CHARS:  # break overlong lines at whitespace
            cut = line.rfind(" ", 0, CHUNK_CHARS) + 1 or CHUNK_CHARS
            lines.append((n, pos, line[:cut]))
            pos, line = pos + cut, line[cut:]
        lines.append((n, pos, line))
        pos += len(line)

    pieces, buf, size = [], [], 0
    for n, at, line in lines:
        heading = line.startswith("#")
        if buf and (size + len(line) > CHUNK_CHARS or (heading and size > CHUNK_CHARS // 2)):
            pieces.append(buf)
            buf, size = [], 0  # carry trailing lines over as overlap, unless a new section starts
            for item in reversed(pieces[-1] if not heading else []):
                if size + len(item[2]) > min(OVERLAP_CHARS, CHUNK_CHARS - len(line)):
                    break
                buf.insert(0, item)
                size += len(item[2])
        buf.append((n, at, line))
        size += len(line)
    pieces.append(buf)

    for piece in filter(None, pieces):  # empty text has an empty piece
        start, end = piece[0][1], piece[-1][1] + len(piece[-1][2])
        body = text[start:end]
        if body.strip():
            start += len(body) - len(body.lstrip())
            yield piece[0][0], piece[-1][0], start, len(body.strip())


def decode(raw):
    """A text file's content: UTF-16 after a UTF-16 byte order mark (Windows PowerShell 5.1 writes those),
    otherwise UTF-8 with invalid bytes replaced."""
    return raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig", "replace")


Doc = namedtuple("Doc", "key digest title meta text chunks")
"""A document read from a file: key is "" for a whole file, the record id for a JSON Lines record; chunks are
(first line, last line, position, length) spans of text."""

ID_FIELDS = ("id", "_id")
TITLE_FIELDS = ("title", "name", "headline", "subject")
TEXT_FIELDS = ("text", "content", "body", "abstract", "description", "summary")


def _digest(raw):
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def _as_text(value):
    """A field's text: a string, or a list of strings joined as paragraphs; "" for anything else."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "\n\n".join(v.strip() for v in value if v.strip())
    return ""


def _record(obj, line, fields):
    """One JSON object as a Doc, or None if it has no text. Without chosen fields, the text is the first of
    TEXT_FIELDS, else the longest string; the title one of TITLE_FIELDS; the id one of ID_FIELDS, else the line
    number. Every field but the text is metadata, as front matter is for Markdown."""
    id_key = fields.get("id") or next((k for k in ID_FIELDS if k in obj), None)
    key = obj.get(id_key)
    key = str(line) if key is None or key == "" else str(key)
    title_key = fields.get("title") or next((k for k in TITLE_FIELDS if _as_text(obj.get(k))), None)
    text_keys = fields.get("text") or [next((k for k in TEXT_FIELDS if _as_text(obj.get(k))), None)]
    if text_keys == [None]:  # no known text field: the longest string is the text
        strings = [(len(v), k) for k, v in obj.items() if k not in (id_key, title_key) and isinstance(v, str)]
        text_keys = [max(strings)[1]] if strings else []
    body = "\n\n".join(t for k in text_keys if (t := _as_text(obj.get(k))))
    if not body:
        return None
    title = _as_text(obj.get(title_key)) if title_key else ""
    text = f"# {title}\n\n{body}" if title else body
    meta = {k: v for k, v in obj.items() if k not in text_keys}
    return Doc(key, "", title or _heading(body) or key, meta, text, list(chunk(text)))


def _records(raw, fields):
    """The documents of a JSON Lines file, one per line, and the lines skipped with their reason."""
    docs, problems, seen = [], [], set()
    for n, line in enumerate(decode(raw).split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line, parse_constant=str)  # NaN/Infinity stay strings: SQLite's JSON rejects them
        except ValueError:
            problems.append(f"line {n}: not JSON")
            continue
        if not isinstance(obj, dict):
            problems.append(f"line {n}: not an object")
            continue
        doc = _record(obj, n, fields)
        if doc is None:
            problems.append(f"line {n}: no text")
        elif doc.key in seen:
            problems.append(f"line {n}: id {doc.key} used twice")
        else:
            seen.add(doc.key)
            docs.append(doc._replace(digest=_digest(line.encode())))
    return docs, problems


def load(path, fields=None):
    """Read one file into documents (runs in worker processes): a Markdown or text file is one document, a JSON
    Lines file one per line. Returns (documents, problems), or None if the file can't be read."""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    if str(path).lower().endswith(JSONL):
        return _records(raw, fields or {})
    text = decode(raw).replace("\r\n", "\n")  # same chunks on every OS
    title, meta, body, first_line = parse(text, Path(path).stem)
    offset = len(text) - len(body)  # the chunks are spans of the whole file, front matter included
    chunks = [(first, last, offset + pos, length) for first, last, pos, length in chunk(body, first_line)]
    return [Doc("", _digest(raw), title, meta, text, chunks)], []


# --- indexing ----------------------------------------------------------------


def _cpus():
    """CPUs this process may use, honouring affinity and container limits where the platform reports them."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):  # Linux
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def workers(files, size=0):
    """Worker processes for parsing `files` files of `size` bytes in all; 0 or 1 means parse in-process.

    One per CPU except the one the main process uses for writing, and at most one per FILES_PER_WORKER files or
    BYTES_PER_WORKER bytes, whichever allows more.
    """
    return max(0, min(_cpus() - 1, max(files // FILES_PER_WORKER, size // BYTES_PER_WORKER)))


def _batches(todo):
    """Files to read per commit: at most WRITE_BATCH files with at most BATCH_BYTES of content."""
    batches, batch, size = [], [], 0
    for item in todo:
        if batch and (len(batch) >= WRITE_BATCH or size + item[3] > BATCH_BYTES):
            batches.append(batch)
            batch, size = [], 0
        batch.append(item)
        size += item[3]
    return [*batches, batch] if batch else batches


def _read(pool, workers, fields, batch):
    """Load a batch of files: pool.map starts all of them at once, builtin map reads one at a time."""
    paths = [item[0] for item in batch]
    if pool:
        return pool.map(load, paths, repeat(fields), chunksize=max(1, min(64, len(paths) // (4 * workers))))
    return map(load, paths, repeat(fields))


def _insert(db, cid, sid, path, doc):
    meta = json.dumps(doc.meta, ensure_ascii=False)
    doc_id = db.execute(
        "INSERT INTO docs(collection, source, path, hash, title, meta, text) VALUES(?,?,?,?,?,?,?)",
        (cid, sid, path, doc.digest, doc.title, meta, doc.text),
    ).lastrowid
    for seq, (first, last, pos, length) in enumerate(doc.chunks):
        c = db.execute(
            "INSERT INTO chunks(doc, seq, line_start, line_end, pos, len) VALUES(?,?,?,?,?,?)",
            (doc_id, seq, first, last, pos, length),
        ).lastrowid
        db.execute("INSERT INTO fts(rowid, title, text) VALUES(?,?,?)", (c, doc.title, doc.text[pos : pos + length]))


def _store(db, cid, sid, rel, docs, counts):
    """Bring one file's documents in the index up to date; unchanged ones (by hash) stay, vectors included."""
    existing = {p: (i, h) for i, p, h in db.execute("SELECT id, path, hash FROM docs WHERE source=?", (sid,))}
    base, kept = rel.rsplit(".", 1)[0], set()
    for doc in docs:
        path = f"{base}/{doc.key}" if doc.key else rel
        kept.add(path)
        old = existing.get(path)
        if old and old[1] == doc.digest:
            continue
        if old:
            delete_docs(db, [old[0]])
        try:
            _insert(db, cid, sid, path, doc)
        except sqlite3.IntegrityError:  # another file of the collection has a document of that name
            print(f"\n{path}: a document of this name exists already, skipped", file=sys.stderr)
            continue
        counts["changed" if old else "added"] += 1
    gone = [i for p, (i, _) in existing.items() if p not in kept]
    delete_docs(db, gone)
    counts["removed"] += len(gone)


def _drop_source(db, sid):
    docs = [d for (d,) in db.execute("SELECT id FROM docs WHERE source=?", (sid,))]
    delete_docs(db, docs)
    db.execute("DELETE FROM sources WHERE id=?", (sid,))
    return len(docs)


def walk(root):
    """Yield (path, mtime_ns, size) for matching files. DirEntry.stat() is free on Windows."""
    stack = [root]
    while stack:
        try:
            entries = os.scandir(stack.pop())
        except OSError:
            continue
        with entries:
            for e in entries:
                if e.is_dir(follow_symlinks=False):
                    if not e.name.startswith(".") and e.name not in SKIP_DIRS:
                        stack.append(e.path)
                elif e.name.lower().endswith(EXTS):
                    try:
                        st = e.stat()
                    except OSError:  # e.g. a broken symlink
                        continue
                    yield e.path, st.st_mtime_ns, st.st_size


def index(db, names=None):
    """Bring the index in sync with the files on disk. Only new or changed files are read, and of those only
    new or changed documents are indexed again."""
    collections = db.execute("SELECT id, name, path, fields FROM collections ORDER BY name").fetchall()
    unknown = set(names or ()) - {name for _, name, _, _ in collections}
    if unknown:
        raise SystemExit(f"no collection named {', '.join(map(repr, sorted(unknown)))}")
    for cid, name, root, fields in collections:
        if names and name not in names:
            continue
        if not os.path.isdir(root):  # never treat an unplugged drive as "all files deleted"
            print(f"{name}: folder not found, skipped: {root}", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        fields = json.loads(fields) if fields else {}
        known = {
            p: (i, m, s)
            for i, p, m, s in db.execute("SELECT id, path, mtime, size FROM sources WHERE collection=?", (cid,))
        }
        todo, seen = [], set()
        for path, mtime, size in walk(root):
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            try:
                rel.encode()
            except UnicodeEncodeError:  # bytes that aren't UTF-8 (Linux); SQLite stores only valid text
                print(f"\nunsupported file name, skipped: {path!r}", file=sys.stderr)
                continue
            seen.add(rel)
            k = known.get(rel)
            if not k or k[1] != mtime or k[2] != size:
                todo.append((path, rel, mtime, size))
        counts = Counter()
        for rel in known.keys() - seen:
            counts["removed"] += _drop_source(db, known[rel][0])

        batches = _batches(todo)
        n = workers(len(todo), sum(item[3] for item in todo))  # parsing is CPU-bound: processes, not threads
        # spawned everywhere, as on Windows and macOS: forking a process that has threads can deadlock
        spawn = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(n, mp_context=spawn) if n > 1 else nullcontext() as pool:
            # double buffering: the next batch is parsed while the current one is written
            pending = _read(pool, n, fields, batches[0]) if batches else None
            done = 0
            for b, batch in enumerate(batches):
                results = pending
                if b + 1 < len(batches):
                    pending = _read(pool, n, fields, batches[b + 1])
                for (path, rel, mtime, size), res in zip(batch, results):
                    if res is None:  # unreadable now; retried on the next run
                        print(f"\nunreadable, skipped: {path}", file=sys.stderr)
                        counts["unreadable"] += 1
                        continue
                    docs, problems = res
                    if problems:
                        shown = "; ".join(problems[:5]) + ("; ..." if len(problems) > 5 else "")
                        lines = "line" if len(problems) == 1 else "lines"
                        print(f"\n{name}: {rel}: {len(problems)} {lines} skipped ({shown})", file=sys.stderr)
                    sid = (
                        known[rel][0]
                        if rel in known
                        else db.execute("INSERT INTO sources(collection, path) VALUES(?, ?)", (cid, rel)).lastrowid
                    )
                    _store(db, cid, sid, rel, docs, counts)
                    db.execute("UPDATE sources SET mtime=?, size=? WHERE id=?", (mtime, size, sid))
                db.commit()
                done += len(batch)
                print(f"\r{name}: {done}/{len(todo)} files read", end="", file=sys.stderr, flush=True)
        db.commit()
        print(
            f"\r{name}: {len(seen)} files, {counts['added']} added, {counts['changed']} changed, "
            f"{counts['removed']} removed"
            + (f", {counts['unreadable']} unreadable" if counts["unreadable"] else "")
            + f" ({time.perf_counter() - t0:.1f}s)",
            file=sys.stderr,
        )
