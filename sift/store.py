"""SQLite store: collections, documents, chunks, keyword index and vectors."""

import hashlib
import json
import multiprocessing
import os
import re
import sqlite3
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

HOME = Path(os.environ.get("SIFT_HOME", Path.home() / ".sift"))
EXTS = (".md", ".markdown", ".txt")
SKIP_DIRS = {"node_modules", "__pycache__"}
CHUNK_CHARS, OVERLAP_CHARS = 3000, 300  # ~750 tokens, well inside the models' 1024-token window
WRITE_BATCH = 5000
FILES_PER_WORKER = 1000  # a worker process only pays off for at least this many files

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS collections(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, collection INTEGER NOT NULL, path TEXT NOT NULL,
    mtime INTEGER, size INTEGER, hash TEXT, title TEXT, meta TEXT, UNIQUE(collection, path));
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, doc INTEGER NOT NULL, seq INTEGER,
    line_start INTEGER, line_end INTEGER);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc);
CREATE TABLE IF NOT EXISTS vectors(chunk INTEGER PRIMARY KEY, vec BLOB NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(title, text, tokenize='unicode61 remove_diacritics 2');
"""


def connect(home=None) -> sqlite3.Connection:
    home = Path(home or HOME)
    home.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(home / "sift.db", timeout=30, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-262144")  # 256 MB
    db.executescript(SCHEMA)
    return db


def get_meta(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, str(value)))


def touch(db):
    """Mark the vectors as changed, so a running server reloads them. Random, as clocks can tick too slowly."""
    set_meta(db, "version", uuid.uuid4().hex)


# --- collections -------------------------------------------------------------


def add_collection(db, path, name=None):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"not a directory: {path}")
    name = name or path.name
    if not re.fullmatch(r"[\w.-]+", name):
        raise SystemExit(f"invalid collection name {name!r}: choose one with --name, of letters, digits, '_', '-', '.'")
    try:
        db.execute("INSERT INTO collections(name, path) VALUES(?, ?)", (name, str(path)))
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
    db.execute("DELETE FROM collections WHERE id=?", row)
    db.commit()
    return len(docs)


def delete_docs(db, doc_ids):
    for i in range(0, len(doc_ids), 500):
        batch = doc_ids[i : i + 500]
        marks = ",".join("?" * len(batch))
        chunk_ids = f"SELECT id FROM chunks WHERE doc IN ({marks})"
        db.execute(f"DELETE FROM fts WHERE rowid IN ({chunk_ids})", batch)
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

    Yields (first line, last line, text).
    """
    lines = []
    for n, line in enumerate(LINE.findall(text), first_line):
        while len(line) > CHUNK_CHARS:  # break overlong lines at whitespace
            cut = line.rfind(" ", 0, CHUNK_CHARS) + 1 or CHUNK_CHARS
            lines.append((n, line[:cut]))
            line = line[cut:]
        lines.append((n, line))

    pieces, buf, size = [], [], 0
    for n, line in lines:
        heading = line.startswith("#")
        if buf and (size + len(line) > CHUNK_CHARS or (heading and size > CHUNK_CHARS // 2)):
            pieces.append(buf)
            buf, size = [], 0  # carry trailing lines over as overlap, unless a new section starts
            for item in reversed(pieces[-1] if not heading else []):
                if size + len(item[1]) > min(OVERLAP_CHARS, CHUNK_CHARS - len(line)):
                    break
                buf.insert(0, item)
                size += len(item[1])
        buf.append((n, line))
        size += len(line)
    pieces.append(buf)

    for piece in pieces:
        body = "".join(text for _, text in piece).strip()
        if body:
            yield piece[0][0], piece[-1][0], body


def decode(raw):
    """A text file's content: UTF-16 after a UTF-16 byte order mark (Windows PowerShell 5.1 writes those),
    otherwise UTF-8 with invalid bytes replaced."""
    return raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig", "replace")


def load(path):
    """Read and parse one file (runs in worker processes). Returns None if unreadable."""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    text = decode(raw).replace("\r\n", "\n")  # same chunks on every OS
    title, meta, body, first_line = parse(text, Path(path).stem)
    return hashlib.blake2b(raw, digest_size=16).hexdigest(), title, meta, list(chunk(body, first_line))


# --- indexing ----------------------------------------------------------------


def _cpus():
    """CPUs this process may use, honouring affinity and container limits where the platform reports them."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):  # Linux
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def workers(files):
    """Worker processes for parsing `files` files; 0 or 1 means parse in-process.

    One per CPU except the one the main process uses for writing, and at most one per FILES_PER_WORKER files.
    """
    return max(0, min(_cpus() - 1, files // FILES_PER_WORKER))


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
    """Bring the keyword index in sync with the files on disk. Only new or changed files are read."""
    collections = db.execute("SELECT id, name, path FROM collections ORDER BY name").fetchall()
    unknown = set(names or ()) - {name for _, name, _ in collections}
    if unknown:
        raise SystemExit(f"no collection named {', '.join(map(repr, sorted(unknown)))}")
    for cid, name, root in collections:
        if names and name not in names:
            continue
        if not os.path.isdir(root):  # never treat an unplugged drive as "all files deleted"
            print(f"{name}: folder not found, skipped: {root}", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        known = {
            p: (i, m, s, h)
            for i, p, m, s, h in db.execute("SELECT id, path, mtime, size, hash FROM docs WHERE collection=?", (cid,))
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
        gone = [known[p][0] for p in known.keys() - seen]
        delete_docs(db, gone)

        added = changed = failed = 0
        batches = [todo[i : i + WRITE_BATCH] for i in range(0, len(todo), WRITE_BATCH)]
        n = workers(len(todo))  # parsing is CPU-bound, so it runs in processes rather than threads
        # spawned everywhere, as on Windows and macOS: forking a process that has threads can deadlock
        spawn = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(n, mp_context=spawn) if n > 1 else nullcontext() as pool:

            def read(batch):  # pool.map starts all work at once; builtin map reads lazily
                paths = [t[0] for t in batch]
                return pool.map(load, paths, chunksize=64) if pool else map(load, paths)

            # double buffering: the next batch is parsed while the current one is written
            pending = read(batches[0]) if batches else None
            for b, batch in enumerate(batches):
                results = pending
                if b + 1 < len(batches):
                    pending = read(batches[b + 1])
                for (path, rel, mtime, size), res in zip(batch, results):
                    if res is None:  # unreadable now; retried on the next run
                        print(f"\nunreadable, skipped: {path}", file=sys.stderr)
                        failed += 1
                        continue
                    digest, title, meta, chunks = res
                    k = known.get(rel)
                    if k and k[3] == digest:  # touched but unchanged
                        db.execute("UPDATE docs SET mtime=?, size=? WHERE id=?", (mtime, size, k[0]))
                        continue
                    if k:
                        delete_docs(db, [k[0]])
                        changed += 1
                    else:
                        added += 1
                    doc = db.execute(
                        "INSERT INTO docs(collection, path, mtime, size, hash, title, meta) VALUES(?,?,?,?,?,?,?)",
                        (cid, rel, mtime, size, digest, title, json.dumps(meta, ensure_ascii=False)),
                    ).lastrowid
                    for seq, (start, end, text) in enumerate(chunks):
                        c = db.execute(
                            "INSERT INTO chunks(doc, seq, line_start, line_end) VALUES(?,?,?,?)", (doc, seq, start, end)
                        ).lastrowid
                        db.execute("INSERT INTO fts(rowid, title, text) VALUES(?,?,?)", (c, title, text))
                db.commit()
                done = min((b + 1) * WRITE_BATCH, len(todo))
                print(f"\r{name}: {done}/{len(todo)} files read", end="", file=sys.stderr, flush=True)
        db.commit()
        print(
            f"\r{name}: {len(seen)} files, {added} added, {changed} changed, {len(gone)} removed"
            + (f", {failed} unreadable" if failed else "")
            + f" ({time.perf_counter() - t0:.1f}s)",
            file=sys.stderr,
        )
