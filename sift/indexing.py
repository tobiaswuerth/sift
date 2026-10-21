"""`sift index`: bringing the index in step with the files on disk, incrementally. Files are compared by modification
time and size, documents by content hash; new and changed files are read in worker processes (see `readers`) while
the previous batch is written.
"""

import json
import multiprocessing
import os
import sqlite3
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from itertools import repeat

from . import console, embed, readers, store

SKIP_DIRS = {"node_modules", "__pycache__"}
WRITE_BATCH = 5000  # files per commit
BATCH_BYTES = 256 << 20  # and at most this much file content, for big JSON Lines files
FILES_PER_WORKER = 1000  # a worker process only pays off for at least this many files
BYTES_PER_WORKER = 32 << 20  # or this much content
PROGRESS_LINES = 200  # at most this many progress updates per collection: printing is slow on Windows consoles


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
        return pool.map(readers.load, paths, repeat(fields), chunksize=max(1, min(64, len(paths) // (4 * workers))))
    return map(readers.load, paths, repeat(fields))


def _insert(db, cid, sid, path, doc):
    """Add a document and its chunks, and return its id. Their keyword entries come with their file's (see _store)."""
    meta = json.dumps(doc.meta, ensure_ascii=False)
    doc_id = db.execute(
        "INSERT INTO docs(collection, source, path, hash, title, meta, text) VALUES(?,?,?,?,?,?,?)",
        (cid, sid, path, doc.digest, doc.title, meta, doc.text),
    ).lastrowid
    db.executemany(
        "INSERT INTO chunks(doc, seq, line_start, line_end, pos, len) VALUES(?,?,?,?,?,?)",
        [(doc_id, seq, first, last, pos, length) for seq, (first, last, pos, length) in enumerate(doc.chunks)],
    )
    return doc_id


def _store(db, cid, sid, rel, docs, counts, note):
    """Bring one file's documents in the index up to date; unchanged ones (by hash) stay, vectors included.
    note(text) tells the user of a document that had to be skipped."""
    existing = {p: (i, h) for i, p, h in db.execute("SELECT id, path, hash FROM docs WHERE source=?", (sid,))}
    base, kept, added = rel.rsplit(".", 1)[0], set(), []
    for doc in docs:
        path = f"{base}/{doc.key}" if doc.key else rel
        kept.add(path)
        old = existing.get(path)
        if old and old[1] == doc.digest:
            continue
        if old:
            store.delete_docs(db, [old[0]])
        try:
            added.append(_insert(db, cid, sid, path, doc))
        except sqlite3.IntegrityError:  # another file of the collection has a document of that name
            note(f"{path}: a document of this name exists already, skipped")
            continue
        counts["changed" if old else "added"] += 1
    store.keywords(db, added)
    gone = [i for p, (i, _) in existing.items() if p not in kept]
    store.delete_docs(db, gone)
    counts["removed"] += len(gone)


REFRESH_BATCH = 1000  # documents read again from the index at a time


def retitle(db, name, title_from):
    """Choose where a collection's Markdown and text files take their titles from (see readers.parse), and retitle
    those indexed from the text the index holds: no file is read again. Returns how many titles changed, and how many
    passages lost their vectors for `sift embed` to embed again."""
    cid, fields = store.set_title_from(db, name, title_from)
    changed = dropped = 0
    for rows in _notes(db, cid):
        c, d = _refresh(db, rows, fields)
        changed, dropped = changed + c, dropped + d
    db.commit()
    return changed, dropped


def _notes(db, cid):
    """The indexed Markdown and text files of a collection, REFRESH_BATCH at a time: (id, path, title, meta, text) of
    each. A file's document is the one with the file's path; JSON Lines files hold records."""
    sql = """SELECT s.id, d.id, d.path, d.title, d.meta, d.text FROM sources s
             JOIN docs d ON d.collection = s.collection AND d.path = s.path AND d.source = s.id
             WHERE s.collection = ? AND s.id > ? ORDER BY s.id LIMIT ?"""
    after = 0
    while rows := db.execute(sql, (cid, after, REFRESH_BATCH)).fetchall():
        after = rows[-1][0]
        yield [row[1:] for row in rows if row[2].lower().endswith(readers.TEXT)]


def _refresh(db, rows, fields):
    """Read notes again from the text the index holds, (id, path, title, meta, text) of each, as this sift reads them
    with the collection's `fields`. Titles and front matter that changed go into the documents and the keyword index;
    passages embedded with a title that changed lose their vectors. Returns how many documents changed, and how many
    vectors went."""
    changed = {}
    for doc_id, path, title, meta, text in rows:
        doc = readers.note(text, path.rsplit("/", 1)[-1], fields)
        stored = json.dumps(doc.meta, ensure_ascii=False)
        if (doc.title, stored) != (title, meta):
            changed[doc_id] = (title, doc.title, stored)
    if not changed:
        return 0, 0
    store.keywords(db, list(changed), add=False)
    db.executemany("UPDATE docs SET title = ?, meta = ? WHERE id = ?", [(t, m, i) for i, (_, t, m) in changed.items()])
    store.keywords(db, list(changed))
    sql = """SELECT c.id, c.doc, c.seq, substr(d.text, c.pos + 1, c.len) FROM chunks c JOIN docs d ON d.id = c.doc
             WHERE c.doc IN (SELECT value FROM json_each(?)) AND c.seq > 0"""
    stale = [
        (chunk,)
        for chunk, doc, seq, text in db.execute(sql, (json.dumps(list(changed)),))
        if embed.text_of(seq, changed[doc][0], text) != embed.text_of(seq, changed[doc][1], text)
    ]
    dropped = db.executemany("DELETE FROM vectors WHERE chunk = ?", stale).rowcount
    if dropped:
        store.touch(db)
    return len(changed), dropped


def _drop_source(db, sid):
    docs = [d for (d,) in db.execute("SELECT id FROM docs WHERE source=?", (sid,))]
    store.delete_docs(db, docs)
    db.execute("DELETE FROM sources WHERE id=?", (sid,))
    return len(docs)


def walk(root):
    """Yield (path, mtime_ns, size) for the files sift has a reader for. DirEntry.stat() is free on Windows."""
    exts, stack = tuple(readers.READERS), [root]
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
                elif e.name.lower().endswith(exts):
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
            console.say(f"{name}: folder not found, skipped: {root}")
            continue
        _index_collection(db, cid, name, root, json.loads(fields) if fields else {})


def _index_collection(db, cid, name, root, fields):
    """Bring one collection in sync: drop what is gone, read what is new or changed."""
    t0 = time.perf_counter()
    known = {
        p: (i, m, s)
        for i, p, m, s in db.execute("SELECT id, path, mtime, size FROM sources WHERE collection=?", (cid,))
    }
    console.say(f"{name}: scanning {root}")  # a large or cloud folder takes a while
    todo, seen = _scan(root, known)
    counts = Counter()
    for rel in known.keys() - seen:
        counts["removed"] += _drop_source(db, known[rel][0])
    if todo:
        with console.Progress(len(todo), "files", prefix=f"{name}: ") as progress:
            _read_and_store(db, cid, name, fields, known, todo, counts, progress)
    db.commit()
    unreadable = f", {counts['unreadable']} unreadable" if counts["unreadable"] else ""
    console.say(
        f"{name}: {len(seen)} files, {counts['added']} added, {counts['changed']} changed, {counts['removed']} removed"
        f"{unreadable} ({time.perf_counter() - t0:.1f}s)"
    )


def _scan(root, known):
    """The files to read, new or changed since `known` (path -> id, mtime, size), and every file seen."""
    todo, seen = [], set()
    for path, mtime, size in walk(root):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        try:
            rel.encode()
        except UnicodeEncodeError:  # bytes that aren't UTF-8 (Linux); SQLite stores only valid text
            console.say(f"unsupported file name, skipped: {path!r}")
            continue
        seen.add(rel)
        k = known.get(rel)
        if not k or k[1] != mtime or k[2] != size:
            todo.append((path, rel, mtime, size))
    return todo, seen


def _read_and_store(db, cid, name, fields, known, todo, counts, progress):
    """Read the files to do, in worker processes when it pays off, and store their documents batch by batch."""
    batches = _batches(todo)
    n = workers(len(todo), sum(item[3] for item in todo))  # parsing is CPU-bound: processes, not threads
    # spawned everywhere, as on Windows and macOS: forking a process that has threads can deadlock
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(n, mp_context=spawn) if n > 1 else nullcontext() as pool:
        # double buffering: the next batch is parsed while the current one is written
        pending = _read(pool, n, fields, batches[0])
        done, step = 0, max(1, len(todo) // PROGRESS_LINES)  # an update per file, or fewer for many small ones
        for b, batch in enumerate(batches):
            results = pending
            if b + 1 < len(batches):
                pending = _read(pool, n, fields, batches[b + 1])
            for (path, rel, mtime, size), res in zip(batch, results):
                done += 1
                if done % step == 0 or done == len(todo):
                    progress.update(done)
                if res is None:  # unreadable now; retried on the next run
                    progress.note(f"unreadable, skipped: {path}")
                    counts["unreadable"] += 1
                    continue
                docs, problems = res
                for problem in problems:
                    progress.note(f"{name}: {rel}: {problem}")
                sid = (
                    known[rel][0]
                    if rel in known
                    else db.execute("INSERT INTO sources(collection, path) VALUES(?, ?)", (cid, rel)).lastrowid
                )
                _store(db, cid, sid, rel, docs, counts, lambda text: progress.note(f"{name}: {text}"))
                db.execute("UPDATE sources SET mtime=?, size=? WHERE id=?", (mtime, size, sid))
            db.commit()
