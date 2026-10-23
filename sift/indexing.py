"""`sift index`: bringing the index in step with the files on disk, incrementally. Files are compared by modification
time and size, documents by content hash; new and changed files are read in worker processes (see `readers`) while
the previous batch is written. `reindex` reads given files at once, for a note just written to be found.
"""

import json
import multiprocessing
import os
import posixpath
import sqlite3
import stat
import time
from collections import Counter, namedtuple
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from itertools import repeat, takewhile
from pathlib import Path

from . import console, embed, links, ocr, readers, store
from .ignore import NAME as SIFTIGNORE
from .ignore import Ignore

SKIP_DIRS = {"node_modules", "__pycache__"}
MOST_FILES = 1000  # files `reindex` reads at once; `sift index` reads any number
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
    """Add a document, its chunks, its links and, if it is a whole file, its names, and return its id. Its keyword
    entries come with its file's (see _store)."""
    meta = json.dumps(doc.meta, ensure_ascii=False)
    doc_id = db.execute(
        "INSERT INTO docs(collection, source, path, hash, title, meta, text) VALUES(?,?,?,?,?,?,?)",
        (cid, sid, path, doc.digest, doc.title, meta, doc.text),
    ).lastrowid
    _add_chunks(db, doc_id, doc.chunks)
    if doc.links or not doc.key:  # a JSON Lines record has no name, and no links: millions of them cost nothing
        links.save(db, [(doc_id, cid, path, doc.meta, doc.links, not doc.key)])
    return doc_id


def _add_chunks(db, doc_id, chunks):
    db.executemany(
        "INSERT INTO chunks(doc, seq, line_start, line_end, pos, len) VALUES(?,?,?,?,?,?)",
        [(doc_id, seq, first, last, pos, length) for seq, (first, last, pos, length) in enumerate(chunks)],
    )


def _path(rel, doc):
    """A document's path in its collection: its file's, or for a record or an attachment, one below it."""
    return f"{rel.rsplit('.', 1)[0]}/{doc.key}" if doc.key else rel


def _read_scans(db, sid, rel, docs, note, now=True):
    """A file's documents with the scans among them read by OCR, those new or changed since the file `sid` was
    indexed (see `ocr`); a scan that can't be read is left out, and noted, as is one read. Reading a scan takes a
    while, so what is written is committed first: no other writer, such as the server's `reindex`, waits for it.
    Unless `now`, a scan to read is left out for `sift index`. Returns the documents, and whether a scan was left."""
    known = dict(db.execute("SELECT path, hash FROM docs WHERE source = ?", (sid,))) if sid else {}
    read, left = [], False
    for doc in docs:
        path = _path(rel, doc)
        if doc.scan is not None and known.get(path) != doc.digest:
            if not now:
                note(f"{path}: a scan, left for `sift index` to read with OCR")
                left = True
                continue
            if db.in_transaction:
                db.commit()
            doc, said = ocr.default().read(doc)
            note(f"{path}: {said}")
            if doc is None:
                continue
        read.append(doc)
    return read, left


def _store(db, cid, sid, rel, docs, counts, note):
    """Bring one file's documents in the index up to date; unchanged ones (by hash) stay, vectors included, and their
    row ids are returned. note(text) tells the user of a document that had to be skipped."""
    existing = {p: (i, h) for i, p, h in db.execute("SELECT id, path, hash FROM docs WHERE source=?", (sid,))}
    kept, added, unchanged = set(), [], set()
    for doc in docs:
        path = _path(rel, doc)
        kept.add(path)
        old = existing.get(path)
        if old and old[1] == doc.digest:
            unchanged.add(old[0])
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
    return unchanged


REFRESH_BATCH = 1000  # files read again from the index at a time


def retitle(db, name, title_from):
    """Choose where a collection's Markdown and text files take their titles from (see readers.parse), and retitle
    those indexed from the text the index holds: no file is read again. Returns how many titles changed, and how many
    passages lost their vectors for `sift embed` to embed again."""
    cid, fields = store.set_title_from(db, name, title_from)
    changed = dropped = 0
    for _, files in _files(db, cid):
        c, d = _reread(db, cid, files, fields)
        changed, dropped = changed + c, dropped + d
    db.commit()
    return changed, dropped


def add_links(db):
    """Read every indexed file's document again from the text the index holds, as this sift reads it, and store its
    links and names: what an index of sift 1.0 lacks. Front matter is read again too, as lists of unquoted items were
    text before. No file is read; nothing is embedded again, unless its title changed (see _reread)."""
    total = db.execute("SELECT count(*) FROM sources").fetchone()[0]
    console.say(f"reading links from the {console.count(total, 'file')} in the index, once ...")
    done = 0
    with console.Progress(total, "files", prefix="links: ") as progress:
        for cid, fields in db.execute("SELECT id, fields FROM collections").fetchall():
            for n, files in _files(db, cid):
                _reread(db, cid, files, json.loads(fields) if fields else {})
                done += n
                progress.update(done)
    console.say(f"links read in {console.duration(progress.elapsed)}")


def _files(db, cid):
    """The documents of a collection's files, REFRESH_BATCH files at a time: how many files, and (id, path, title,
    meta) of their documents. A file's document has the file's path; a JSON Lines file has records instead."""
    sql = """SELECT s.id, d.id, d.path, d.title, d.meta FROM sources s
             LEFT JOIN docs d ON d.collection = s.collection AND d.path = s.path AND d.source = s.id
             WHERE s.collection = ? AND s.id > ? ORDER BY s.id LIMIT ?"""
    after = 0
    while rows := db.execute(sql, (cid, after, REFRESH_BATCH)).fetchall():
        after = rows[-1][0]
        yield len(rows), [row[1:] for row in rows if row[1] is not None]


def _reread(db, cid, files, fields):
    """Read the documents of files again from the index, (id, path, title, meta) of each: Markdown and text files from
    the text the index holds, as this sift reads them with the collection's `fields`, other files only for their
    names. Their names and links are stored anew; titles and front matter that changed go into the documents and the
    keyword index, and passages embedded with a title that changed lose their vectors. A note of front matter alone
    gets its passage (sift 1.0 gave it none). Returns how many documents changed, and how many vectors went."""
    notes = json.dumps([doc for doc, path, _, _ in files if path.lower().endswith(readers.TEXT)])
    texts = dict(db.execute("SELECT id, text FROM docs WHERE id IN (SELECT value FROM json_each(?))", (notes,)))
    sql = "SELECT doc, count(*) FROM chunks WHERE doc IN (SELECT value FROM json_each(?)) GROUP BY doc"
    passages = dict(db.execute(sql, (notes,)))
    changed, fresh, saved = {}, {}, []
    for doc_id, path, title, meta in files:
        if doc_id not in texts:
            saved.append((doc_id, cid, path, json.loads(meta), (), True))
            continue
        doc = readers.note(texts[doc_id], path.rsplit("/", 1)[-1], fields)
        saved.append((doc_id, cid, path, doc.meta, doc.links, True))
        stored = json.dumps(doc.meta, ensure_ascii=False)
        if (doc.title, stored) != (title, meta):
            changed[doc_id] = (title, doc.title, stored)
        if doc.chunks and doc_id not in passages:
            fresh[doc_id] = doc.chunks
    ids = json.dumps([doc for doc, *_ in files])
    for table in ("links", "names"):
        db.execute(f"DELETE FROM {table} WHERE doc IN (SELECT value FROM json_each(?))", (ids,))
    links.save(db, saved)
    store.keywords(db, list(changed), add=False)
    db.executemany("UPDATE docs SET title = ?, meta = ? WHERE id = ?", [(t, m, i) for i, (_, t, m) in changed.items()])
    for doc_id, chunks in fresh.items():
        _add_chunks(db, doc_id, chunks)
    store.keywords(db, list(changed.keys() | fresh.keys()))
    if not changed:
        return 0, 0
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


Reindexed = namedtuple("Reindexed", "files removed before after")
"""What `reindex` did: per file, ({"id", "status"[, "note"]}, its documents' row ids); the chunks it removed; and the
index's version before and after, for a server to follow without loading every vector again."""


def reindex(db, paths, scans=True):
    """Read files again now, or remove their documents if they are gone, so that the next search finds them as they
    are: their text, front matter, links and keyword entries. `paths` are ids, `collection/path`, or absolute paths in
    a collection, each a file `sift index` would read; if one isn't, nothing is read. A file's status is "added",
    "changed", "unchanged", "removed", "not found", "skipped" (no documents in it, with a "note" saying why) or
    "unreadable". In one transaction. Without `scans`, scans to read with OCR are left for `sift index`, which reads
    their files again even if they don't change."""
    if not isinstance(paths, list | tuple) or not all(isinstance(path, str) for path in paths):
        raise ValueError(f"paths must be a list of ids or absolute paths, got {paths!r}")
    if not paths:
        raise ValueError("paths needs at least one id or absolute path")
    if len(paths) > MOST_FILES:
        raise ValueError(f"at most {MOST_FILES:,} files at a time; `sift index` reads any number")
    files = list(dict.fromkeys(_locate(db, path) for path in paths))
    on_disk = [
        _reindex_read(db, cid, root, json.loads(fields) if fields else {}, rel, scans)
        for cid, _, root, fields, rel in files
    ]
    db.execute("BEGIN IMMEDIATE")
    try:
        before, found, removed = store.get_meta(db, "version"), [], []
        for (cid, name, _, _, rel), found_on_disk in zip(files, on_disk):
            entry, docs, gone = _reindex_file(db, cid, name, rel, *found_on_disk)
            found.append((entry, docs))
            removed += gone
        after = store.get_meta(db, "version")
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return Reindexed(found, removed, before, after)


def _locate(db, path):
    """The file an id or absolute path names, as `sift index` stores it: (collection id, name, folder, fields, path in
    it), in the file system's spelling. A file `sift index` wouldn't read is an error."""
    collections = db.execute("SELECT id, name, path, fields FROM collections ORDER BY length(path) DESC").fetchall()
    given = os.path.expanduser(path)
    if os.path.isabs(given):
        full = os.path.normpath(given)
        inside = [c for c in collections if os.path.normcase(full).startswith(os.path.normcase(os.path.join(c[2], "")))]
        if not inside:
            raise ValueError(_nowhere(path, full, collections))
        (cid, name, root, fields), rel = inside[0], os.path.relpath(full, inside[0][2]).replace(os.sep, "/")
    else:
        name, _, rel = path.replace("\\", "/").partition("/")
        match = [c for c in collections if c[1] == name]
        if not match:
            raise store.no_collection(db, name)
        (cid, name, root, fields), rel = match[0], posixpath.normpath(rel) if rel else ""
        if rel in ("", "."):
            raise ValueError(f"{path!r}: name a file of the collection, as collection/path, or by its absolute path")
        if rel == ".." or rel.startswith(("../", "/")):
            raise ValueError(f"{path!r} is outside the collection {name!r}")
    if not os.path.isdir(root):  # never take an unplugged drive for files deleted
        raise ValueError(f"{name}: folder not found: {root}")
    parts = _spelled(root, rel.split("/"))
    for part in parts[:-1]:
        if part.startswith(".") or part in SKIP_DIRS:
            raise ValueError(f"{name}/{rel}: sift indexes no files in {part}/")
    try:
        ignore = Ignore.of(root)
    except OSError as error:
        raise ValueError(f"{name}: its {SIFTIGNORE} can't be read, so no file was reindexed: {error}") from None
    if ignore.excludes("/".join(parts)):
        raise ValueError(f"{name}/{rel}: the collection's {SIFTIGNORE} leaves it out")
    ext = readers.extension(parts[-1])
    if ext not in readers.READERS:
        raise ValueError(f"{name}/{rel}: sift doesn't index {f'{ext} files' if ext else 'files without an extension'}")
    for i in range(1, len(parts)):
        if os.path.islink(os.path.join(root, *parts[:i])):
            raise ValueError(
                f"{name}/{rel}: {parts[i - 1]}/ is linked from elsewhere, and `sift index` doesn't follow it"
            )
    rel = "/".join(parts)
    try:
        rel.encode()
    except UnicodeEncodeError:  # bytes that aren't UTF-8 (Linux); SQLite stores only valid text
        raise ValueError(f"{name}: unsupported file name {rel!r}") from None
    if not os.path.lexists(os.path.join(root, *parts)):  # gone: the file as indexed, in whatever spelling
        rel = next(iter(_indexed_as(db, cid, rel)), rel)
    return cid, name, root, fields, rel


NEAREST_SHOWN = 5  # collections an error about a file in none names, the nearest first


def _nowhere(path, full, collections):
    """Why the file at `full` is in no collection, and how it could be: the collections' folders, nearest first, and
    the `sift add` that would cover it."""
    parts = Path(os.path.normcase(full)).parts

    def shared(root):  # how many folders, from the drive or root on, the file and the folder have in common
        return len(list(takewhile(lambda pair: pair[0] == pair[1], zip(parts, Path(os.path.normcase(root)).parts))))

    nearest = sorted(collections, key=lambda c: (-shared(c[2]), c[1]))
    listing = "; ".join(f"{name}: {root}" for _, name, root, _ in nearest[:NEAREST_SHOWN])
    listing += "; ..." if len(nearest) > NEAREST_SHOWN else ""
    head = f"The collections and their folders: {listing}." if collections else "none is registered yet."
    return (
        f"{path} is in no collection{'.' if collections else ':'} {head} To index files there, add their folder as a "
        f"collection: `{console.command('sift', 'add', os.path.dirname(full))}`"
    )


def _spelled(root, parts):
    """A path's parts as the file system spells them: on Windows and macOS a path names a file in any case or Unicode
    form, which indexed so would be the file a second time. Parts that don't exist stay as given."""
    folder, spelled = root, []
    for i, part in enumerate(parts):
        try:
            names = os.listdir(folder)
        except OSError:
            return spelled + parts[i:]
        if part not in names:
            same = [n for n in names if readers.fold(n) == readers.fold(part)]
            part = same[0] if len(same) == 1 else part
        spelled.append(part)
        folder = os.path.join(folder, part)
    return spelled


def _indexed_as(db, cid, rel):
    """The paths of the collection's files that `rel` names, in any case or Unicode form: one, or none."""
    if db.execute("SELECT 1 FROM sources WHERE collection = ? AND path = ?", (cid, rel)).fetchone():
        return [rel]
    paths = [p for (p,) in db.execute("SELECT path FROM sources WHERE collection = ?", (cid,))]
    same = [p for p in paths if readers.fold(p) == readers.fold(rel)]
    return same if len(same) == 1 else []


def _reindex_read(db, cid, root, fields, rel, scans):
    """A file as `reindex` finds it, before anything is written: its stat, None if it is gone; its documents and
    problems as `sift index` reads them, scans read if `scans`, None if it can't be read; and whether a scan was left
    for `sift index`."""
    path = os.path.join(root, *rel.split("/"))
    try:
        info = os.stat(path)  # through a link, as `sift index` reads it
    except OSError:
        info = None
    if info is None or not stat.S_ISREG(info.st_mode):
        return None, None, False
    read = readers.load(path, fields)
    if read is None:
        return info, None, False
    docs, problems = read
    known = db.execute("SELECT id FROM sources WHERE collection = ? AND path = ?", (cid, rel)).fetchone()
    docs, left = _read_scans(db, known and known[0], rel, docs, problems.append, now=scans)
    return info, (docs, problems), left


def _reindex_file(db, cid, name, rel, info, read, left):
    """Bring one file's documents up to date with what `_reindex_read` found, or remove them if it is gone, as `sift
    index` would: ({"id", "status"}, its documents' row ids, the chunks removed). A chunk removed is every chunk of a
    document replaced or removed: SQLite hands the ids of deleted chunks out again, so the new chunks of a note may
    carry its old ones' ids. A file with a scan `left` is read again by the next `sift index`."""
    entry = {"id": f"{name}/{rel}"}
    known = db.execute("SELECT id FROM sources WHERE collection = ? AND path = ?", (cid, rel)).fetchone()
    sid = known and known[0]
    sql = "SELECT c.doc, c.id FROM chunks c JOIN docs d ON d.id = c.doc WHERE d.source = ?"
    chunks = db.execute(sql, (sid,)).fetchall()
    if info is None:
        if sid is None:
            return entry | {"status": "not found"}, [], []
        _drop_source(db, sid)
        return entry | {"status": "removed"}, [], [chunk for _, chunk in chunks]
    if read is None:
        return entry | {"status": "unreadable"}, [], []
    docs, problems = read
    if sid is None:
        sid = db.execute("INSERT INTO sources(collection, path) VALUES(?, ?)", (cid, rel)).lastrowid
    had, counts = db.execute("SELECT count(*) FROM docs WHERE source = ?", (sid,)).fetchone()[0], Counter()
    unchanged = _store(db, cid, sid, rel, docs, counts, problems.append)
    mtime = None if left else info.st_mtime_ns  # no time: `sift index` takes it for changed
    db.execute("UPDATE sources SET mtime = ?, size = ? WHERE id = ?", (mtime, info.st_size, sid))
    now = [d for (d,) in db.execute("SELECT id FROM docs WHERE source = ?", (sid,))]
    if not now and problems:
        status = "skipped"
    elif not (counts["added"] or counts["changed"] or counts["removed"]):
        status = "unchanged"
    else:
        status = "changed" if had else "added"
    note = {"note": "; ".join(problems)} if problems else {}
    return entry | {"status": status} | note, now, [chunk for doc, chunk in chunks if doc not in unchanged]


def _drop_source(db, sid):
    docs = [d for (d,) in db.execute("SELECT id FROM docs WHERE source=?", (sid,))]
    store.delete_docs(db, docs)
    db.execute("DELETE FROM sources WHERE id=?", (sid,))
    return len(docs)


def walk(root, ignore=None):
    """Yield (path, mtime_ns, size) for the files sift has a reader for, but those in hidden folders, SKIP_DIRS and
    what `ignore`, the collection's .siftignore, leaves out. DirEntry.stat() is free on Windows."""
    stack = [(root, "")]
    while stack:
        folder, rel = stack.pop()
        try:
            entries = os.scandir(folder)
        except OSError:
            continue
        with entries:
            for e in entries:
                if e.is_dir(follow_symlinks=False):
                    if (
                        not e.name.startswith(".")
                        and e.name not in SKIP_DIRS
                        and not (ignore and ignore(rel + e.name, True))
                    ):
                        stack.append((e.path, rel + e.name + "/"))
                elif readers.extension(e.name) in readers.READERS and not (ignore and ignore(rel + e.name)):
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
        try:
            ignore = Ignore.of(root)
        except OSError as error:  # indexing all would take in the very files it keeps out
            console.say(f"{name}: skipped, as its {SIFTIGNORE} can't be read: {error}")
            continue
        _index_collection(db, cid, name, root, json.loads(fields) if fields else {}, ignore)


def _index_collection(db, cid, name, root, fields, ignore):
    """Bring one collection in sync: drop what is gone, read what is new or changed."""
    t0 = time.perf_counter()
    known = {
        p: (i, m, s)
        for i, p, m, s in db.execute("SELECT id, path, mtime, size FROM sources WHERE collection=?", (cid,))
    }
    console.say(f"{name}: scanning {root}")  # a large or cloud folder takes a while
    todo, seen = _scan(root, known, ignore)
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


def _scan(root, known, ignore):
    """The files to read, new or changed since `known` (path -> id, mtime, size), and every file seen but those
    `ignore` leaves out."""
    todo, seen = [], set()
    for path, mtime, size in walk(root, ignore):
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
                note = lambda text: progress.note(f"{name}: {text}")  # noqa: E731
                docs, _ = _read_scans(db, known[rel][0] if rel in known else None, rel, docs, note)
                sid = (
                    known[rel][0]
                    if rel in known
                    else db.execute("INSERT INTO sources(collection, path) VALUES(?, ?)", (cid, rel)).lastrowid
                )
                _store(db, cid, sid, rel, docs, counts, note)
                db.execute("UPDATE sources SET mtime=?, size=? WHERE id=?", (mtime, size, sid))
            db.commit()
