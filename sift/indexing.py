"""`sift index`: bringing the index in step with the files on disk, incrementally. Files are compared by modification
time and size, documents by content hash; new and changed files are read in worker processes (see `readers`) while
the previous batch is written.
"""

import json
import multiprocessing
import os
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from itertools import repeat

from . import readers, store

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
            store.delete_docs(db, [old[0]])
        try:
            _insert(db, cid, sid, path, doc)
        except sqlite3.IntegrityError:  # another file of the collection has a document of that name
            print(f"\n{path}: a document of this name exists already, skipped", file=sys.stderr)
            continue
        counts["changed" if old else "added"] += 1
    gone = [i for p, (i, _) in existing.items() if p not in kept]
    store.delete_docs(db, gone)
    counts["removed"] += len(gone)


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
            print(f"{name}: folder not found, skipped: {root}", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        fields = json.loads(fields) if fields else {}
        known = {
            p: (i, m, s)
            for i, p, m, s in db.execute("SELECT id, path, mtime, size FROM sources WHERE collection=?", (cid,))
        }
        todo, seen = [], set()
        print(f"{name}: scanning {root}", file=sys.stderr, flush=True)  # a large or cloud folder takes a while
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
            done, step = 0, max(1, len(todo) // PROGRESS_LINES)  # a line per file, or fewer for many small ones
            for b, batch in enumerate(batches):
                results = pending
                if b + 1 < len(batches):
                    pending = _read(pool, n, fields, batches[b + 1])
                for (path, rel, mtime, size), res in zip(batch, results):
                    done += 1
                    if done % step == 0 or done == len(todo):
                        print(f"\r{name}: {done}/{len(todo)} files read", end="", file=sys.stderr, flush=True)
                    if res is None:  # unreadable now; retried on the next run
                        print(f"\nunreadable, skipped: {path}", file=sys.stderr)
                        counts["unreadable"] += 1
                        continue
                    docs, problems = res
                    for problem in problems:
                        print(f"\n{name}: {rel}: {problem}", file=sys.stderr)
                    sid = (
                        known[rel][0]
                        if rel in known
                        else db.execute("INSERT INTO sources(collection, path) VALUES(?, ?)", (cid, rel)).lastrowid
                    )
                    _store(db, cid, sid, rel, docs, counts)
                    db.execute("UPDATE sources SET mtime=?, size=? WHERE id=?", (mtime, size, sid))
                db.commit()
        db.commit()
        print(
            f"\r{name}: {len(seen)} files, {counts['added']} added, {counts['changed']} changed, "
            f"{counts['removed']} removed"
            + (f", {counts['unreadable']} unreadable" if counts["unreadable"] else "")
            + f" ({time.perf_counter() - t0:.1f}s)",
            file=sys.stderr,
        )
