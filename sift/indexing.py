"""`sift index`: bringing the index in step with the files on disk, incrementally. Files are compared by modification
time and size, documents by content hash; new and changed files are read in worker processes (see `readers`) while
the previous batch is written. `reindex` reads given files at once, for a note just written to be found.
"""

import contextlib
import json
import multiprocessing
import os
import posixpath
import sqlite3
import stat
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
from itertools import takewhile
from pathlib import Path
from typing import Literal, NamedTuple, TypeGuard, TypeVar, overload

from . import console, embed, links, ocr, postings, readers, store
from .ignore import NAME as SIFTIGNORE
from .ignore import Ignore

SKIP_DIRS = {"node_modules", "__pycache__"}
GONE = (FileNotFoundError, NotADirectoryError)  # a path that isn't there any more; other errors: there, out of reach
MOST_FILES = 1000  # files `reindex` reads at once; `sift index` reads any number
WRITE_BATCH = 5000  # files per commit
BATCH_BYTES = 256 << 20  # and at most this much file content, for big JSON Lines files
FILES_PER_WORKER = 1000  # a worker process only pays off for at least this many files
BYTES_PER_WORKER = 32 << 20  # or this much content
PROGRESS_LINES = 200  # at most this many progress updates per collection: printing is slow on Windows consoles
WINDOWS = os.name == "nt"  # where "C:x.md", "sub./x.md" and "x.md:stream" name other files, or none

Db = sqlite3.Connection
Catalog = store.Catalog
C = TypeVar("C", bound=store.Collection)


Fields = readers.Fields
Item = tuple[str, str, int, int]  # a file to read: its path, its path in the collection, its mtime_ns and size
Loaded = tuple[list[readers.Doc], list[str]] | None | Exception  # what readers.load gives, or the error a reader raised
Note = Callable[[str], None]
Twice = Callable[[readers.Doc], None] | None
Counts = Counter[str]
Unlisted = list[tuple[str, OSError]]
Task = tuple[list[tuple[str, int]], Future[list[Loaded]], ProcessPoolExecutor]
FileRow = tuple[int, str, str, str]  # a document's id, path, title and meta, as stored
# a file's stat, the error that kept it from being stat'ed, or None if it is gone
Info = os.stat_result | OSError | None
Entry = dict[str, str]  # {"id", "status"[, "note"]}


def cpus() -> int:
    """CPUs this process may use, honouring affinity and container limits where the platform reports them."""
    process_cpu_count: Callable[[], int | None] | None = getattr(os, "process_cpu_count", None)  # Python 3.13+
    if process_cpu_count:
        return process_cpu_count() or 1
    sched_getaffinity: Callable[[int], set[int]] | None = getattr(os, "sched_getaffinity", None)  # Linux
    if sched_getaffinity:
        return len(sched_getaffinity(0))
    return os.cpu_count() or 1


def workers(files: int, size: int = 0) -> int:
    """Worker processes for parsing `files` files of `size` bytes in all; 0 or 1 means parse in-process.

    One per CPU except the one the main process uses for writing, and at most one per FILES_PER_WORKER files or
    BYTES_PER_WORKER bytes, whichever allows more.
    """
    return max(0, min(cpus() - 1, max(files // FILES_PER_WORKER, size // BYTES_PER_WORKER)))


def _batches(todo: Iterable[Item]) -> list[list[Item]]:
    """Files to read per commit: at most WRITE_BATCH files with at most BATCH_BYTES of content."""
    batches: list[list[Item]] = []
    batch: list[Item] = []
    size = 0
    for item in todo:
        if batch and (len(batch) >= WRITE_BATCH or size + item[3] > BATCH_BYTES):
            batches.append(batch)
            batch, size = [], 0
        batch.append(item)
        size += item[3]
    return [*batches, batch] if batch else batches


def _whole(todo: Iterable[Item]) -> list[Item]:
    """The files to do that are read whole, each before it is stored, not as it is stored (readers.Records)."""
    return [item for item in todo if not readers.streamed(item[0], item[3])]


def _read(pool: "_Workers | None", fields: Fields, batch: list[Item]) -> Callable[[], list[Loaded | readers.Records]]:
    """Start loading a batch of files: in the worker processes of `pool` all at once, without one none yet. Returns
    what gives them, in the batch's order, when called: a file read as it is stored as its Records, unread, here, as
    workers hand over what they read whole."""
    whole = [(item[0], item[3]) for item in _whole(batch)]
    running = (pool, pool.start(whole)) if pool else None

    def loaded() -> list[Loaded | readers.Records]:
        read = iter(running[0].results(running[1]) if running else _loaded_all(_loaded, [p for p, _ in whole], fields))
        streamed = readers.streamed
        return [readers.Records(path, fields) if streamed(path, size) else next(read) for path, _, _, size in batch]

    return loaded


def _loaded(path: str, fields: Fields) -> Loaded:
    """What readers.load reads of a file, or the error a reader raised (runs in worker processes): one file's
    trouble, a record of gigabytes, a reader's bug, mustn't stop the others, nor every later run."""
    try:
        return readers.load(path, fields)
    except Exception as error:
        return error


def _loaded_all(read: Callable[[str, Fields], Loaded], paths: list[str], fields: Fields) -> list[Loaded]:
    """What `read` gives of each of a task's files (runs in worker processes)."""
    return [read(path, fields) for path in paths]


READ_SECONDS = 600  # a process gets this long for the files of a task before it is taken for hung: a long PDF takes
READ_RATE = 100_000  # minutes; and a second more for every so many bytes of them, 43 minutes more for 256 MB


class Stopped(Exception):
    """A file whose reader was stopped: it ended the process it ran in, or came to no end."""


def _patience(files: Iterable[tuple[str, int]]) -> float:
    """Seconds reading these files, (path, size) each, may take before their process is taken for hung."""
    return READ_SECONDS + sum(size for _, size in files) / READ_RATE


def _pool(n: int) -> ProcessPoolExecutor:
    """`n` worker processes. Spawned everywhere, as on Windows and macOS: forking a process that has threads can
    deadlock."""
    return ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("spawn"))


def _end(pool: ProcessPoolExecutor) -> None:
    """End a pool at once: its processes are killed, one that hangs too, and what was started in it is lost. Before
    Python 3.14 (kill_workers) a pool names its processes here alone."""
    for process in list(pool._processes.values()):
        process.kill()
    pool.shutdown(wait=False, cancel_futures=True)


class _Workers:
    """Worker processes that read files, several to a task (see start), handed over in the order started. A reader's
    native code can end its process, or never come to an end: every task not done then goes with the pool, which
    tells no more than that it broke. Their files are read again one at a time, in a process of their own, so that
    the one that ends or hangs it is known: it is left out, as Stopped, and the others are read. Files started in a
    pool that those of another batch ended start again in the next one."""

    single: ProcessPoolExecutor | None

    def __init__(self, n: int, fields: Fields) -> None:
        self.n, self.fields, self.pool, self.single = n, fields, _pool(n), None

    def __enter__(self) -> "_Workers":
        return self

    def __exit__(self, kind: type[BaseException] | None, *exc: object) -> None:
        for pool in filter(None, (self.pool, self.single)):
            if kind is None:
                pool.shutdown()
            else:  # stopped, as by Ctrl+C: not once every file started is read
                _end(pool)

    def start(self, files: list[tuple[str, int]]) -> list[Task]:
        """Start reading these files, (path, size) each, some to a task: fewer tasks cost less to hand over. Returns
        what `results` takes."""
        size = max(1, min(64, len(files) // (4 * self.n)))
        return [self._task(self.pool, files[i : i + size]) for i in range(0, len(files), size)]

    def _task(self, pool: ProcessPoolExecutor, files: list[tuple[str, int]]) -> Task:
        return files, pool.submit(_loaded_all, _loaded, [path for path, _ in files], self.fields), pool

    def results(self, started: list[Task]) -> list[Loaded]:
        """What each of the files started was read as, in order: as _loaded gives it, or Stopped."""
        read: list[Loaded] = []
        at = 0
        while at < len(started):
            files, future, pool = started[at]
            try:
                read += future.result(timeout=_patience(files))
                at += 1
            except (BrokenProcessPool, CancelledError, TimeoutError):
                if pool is self.pool:  # a file not read yet ended it, or hangs: which of them, each alone tells
                    _end(pool)
                    self.pool = _pool(self.n)
                    return read + [self._alone(file) for files, _, _ in started[at:] for file in files]
                started[at:] = [self._task(self.pool, files) for files, _, _ in started[at:]]  # another batch's did
        return read

    def _alone(self, file: tuple[str, int]) -> Loaded:
        """What one file is read as in a process that reads nothing else, or Stopped."""
        self.single = self.single or _pool(1)
        patience = _patience([file])
        try:
            return self._task(self.single, [file])[1].result(timeout=patience)[0]
        except (BrokenProcessPool, TimeoutError) as ended:
            _end(self.single)
            self.single = None
            if isinstance(ended, TimeoutError):
                return Stopped(f"its reader came to no end in {console.duration(patience)}")
            return Stopped("its reader ended the process it ran in")


def _storable(text: str) -> str:
    """Text as the store keeps it: a NUL, at which SQLite's substr stops cutting passages out of the text, and with it
    what is searched, and half a surrogate pair, as UTF-7 can encode, which SQLite's UTF-8 can't hold, as U+FFFD, of
    the same length, so that the chunks' positions stay."""
    return readers.LONE.sub("\ufffd", text.replace("\x00", "\ufffd"))


def _insert(db: Db, sid: int, path: str, doc: readers.Doc) -> int:
    """Add a document, its chunks, its links and, if it is a whole file, its names, and return its id. Its keyword
    entries come with its file's (see _store)."""
    title, meta, text = map(_storable, (doc.title, json.dumps(doc.meta, ensure_ascii=False), doc.text))
    doc_id = store.inserted(
        db.execute(
            "INSERT INTO docs(source, path, hash, title, meta, text) VALUES(?,?,?,?,?,?)",
            (sid, path, doc.digest, title, meta, text),
        )
    )
    _add_chunks(db, doc_id, doc.chunks)
    if doc.links or not doc.key:  # a JSON Lines record has no name, and no links: millions of them cost nothing
        links.save(db, [(doc_id, path, doc.meta, doc.links, not doc.key)])
    return doc_id


def _add_chunks(db: Db, doc_id: int, chunks: Iterable[readers.Chunk]) -> None:
    db.executemany(
        "INSERT INTO chunks(doc, seq, line_start, line_end, pos, len) VALUES(?,?,?,?,?,?)",
        [(doc_id, seq, first, last, pos, length) for seq, (first, last, pos, length) in enumerate(chunks)],
    )


def _path(rel: str, doc: readers.Doc) -> str:
    """A document's path in its collection: its file's, or for a record or an attachment, one below it, its key as the
    store keeps text (a JSON Lines id or a mail's attachment name may hold a NUL)."""
    return f"{rel.rsplit('.', 1)[0]}/{_storable(doc.key)}" if doc.key else rel


SCANS_SKIPPED = "scans_skipped"  # in a store's meta table: the SIFT_OCR under which a scan was last left unread
# How far each SIFT_OCR reads: on, every scan auto reads; auto, every one off does
OCR_READS = {"off": 0, "auto": 1, "on": 2}


def _read_skipped_scans(db: Db) -> None:
    """Have this index read the scans left unread under a SIFT_OCR that reads fewer than the one now (SCANS_SKIPPED),
    as it was turned on, or from off to auto: the PDFs without documents, and the mails with one attached, once. A
    skipped scan's file keeps its time, so that no later run reads it again only to skip it again."""
    skipped = store.get_meta(db, SCANS_SKIPPED)
    if skipped is not None and OCR_READS[ocr.setting()] > OCR_READS[skipped]:
        store.read_again(db, (".pdf",))
        db.execute("DELETE FROM meta WHERE key = ?", (SCANS_SKIPPED,))
        db.commit()


def _read_scans(
    db: Db, sid: int | None, rel: str, docs: Iterable[readers.Doc], note: Note, now: bool = True
) -> tuple[list[readers.Doc], bool]:
    """A file's documents with the scans among them read by OCR, those new or changed since the file `sid` was
    indexed (see `ocr`); a scan that can't be read is left out, and noted, as is one read. Reading a scan takes a
    while, so what is written is committed first: no other writer, such as the server's `reindex`, waits for it.
    Unless `now`, a scan to read is left out for `sift index`. Returns the documents, and whether a scan was left,
    for now or for a try that may read it (its GPU busy, OCR failing): its file is read again by the next index."""
    known = dict(db.execute("SELECT path, hash FROM docs WHERE source = ?", (sid,))) if sid else {}
    read: list[readers.Doc] = []
    left = False
    for doc in docs:
        path = _path(rel, doc)
        if doc.scan is not None and known.get(path) != doc.digest:
            if not now:
                note(f"{path}: a scan, left for `sift index` to read with OCR")
                left = True
                continue
            if db.in_transaction:
                db.commit()
            doc, said, again = ocr.default().read(doc, doc.scan)
            note(f"{path}: {said}")
            if doc is None:
                left |= again
                if not again and ocr.default().unavailable() is not None:  # SIFT_OCR's doing, not the scan's
                    store.set_meta(db, SCANS_SKIPPED, ocr.setting())
                    db.commit()
                continue
        read.append(doc)
    return read, left


STORED = 10_000  # documents of a file stored at a time,
STORED_CHARS = 16 << 20  # of this much text at most, or one document: what is in memory of a file read as it is stored
DROPPED = 10_000  # documents a file no longer holds, deleted at a time


def _batches_of(docs: Iterable[readers.Doc]) -> Iterator[list[readers.Doc]]:
    """A file's documents in the batches they are stored in: STORED of them, of STORED_CHARS of text or one alone."""
    batch: list[readers.Doc] = []
    size = 0
    for doc in docs:
        batch.append(doc)
        size += len(doc.text)
        if len(batch) >= STORED or size >= STORED_CHARS:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def _store(
    db: Db, sid: int, rel: str, docs: Iterable[readers.Doc], counts: Counts, note: Note, twice: Twice = None
) -> None:
    """Bring one file's documents in the index up to date, a batch at a time (see _batches_of), so that a file read
    as it is stored is in memory a batch at a time. Unchanged ones (by hash) stay, vectors included; those the file
    no longer holds go. note(text) tells the user of a document left out (see _left_out). A document whose path the
    file gave before is left out, as the first keeps it: twice(doc) is told, if given, as a file read as it is stored
    can't tell itself (readers.Records)."""
    store.stop_waiting(db, sid)  # what still waits is said again below
    indexed = _Indexed(db, sid)
    for batch in _batches_of(docs):
        paths = [_path(rel, doc) for doc in batch]
        held = indexed.at(paths)
        added: list[int] = []
        unchanged: list[int] = []
        given: set[str] = set()
        chunks = 0
        replaced: list[int] = []  # old versions of documents that changed
        new: list[tuple[str, readers.Doc, bool]] = []  # (path, doc, changed?) to insert
        again: list[tuple[str, readers.Doc]] = []
        for path, doc in zip(paths, batch):
            old = held.get(path)
            if path in given or (old and old[2]):
                again.append((path, doc))  # once the first is in, which keeps the path
                continue
            given.add(path)
            if old and old[1] == doc.digest:
                unchanged.append(old[0])
                continue
            if old:
                replaced.append(old[0])
            new.append((path, doc, bool(old)))
        # Together, not one by one, which cost twice the statements: 50,000 changed records took 11.4 s to store, not
        # 5.7. Their new versions take the paths at once.
        store.delete_docs(db, replaced, free=False)
        for path, doc, changed in new:
            try:
                added.append(_insert(db, sid, path, doc))
            except sqlite3.IntegrityError:  # another file's document holds the path
                _left_out(db, sid, rel, path, note)
                continue
            counts["changed" if changed else "added"] += 1
            chunks += len(doc.chunks)
        for path, doc in again:
            if twice:
                twice(doc)
            else:
                _left_out(db, sid, rel, path, note)
        store.keywords(db, added, count=chunks)
        indexed.took(given, unchanged, added)
    counts["removed"] += indexed.drop_the_gone()


class _Indexed:
    """One file's documents as the index holds them, while the file is stored anew: those at a batch's paths (at),
    what a batch did with them (took), and at the end those the file no longer holds (drop_the_gone). STORED of them
    at most are held here, which costs a small file one statement. A file of more has them found by their paths, a
    batch at a time, not among all of its own, which every batch would read through; and those that stay as they are
    are kept in a table of the connection's own, emptied at the end: of millions of records, too many to hold."""

    STAYING = "CREATE TEMP TABLE IF NOT EXISTS sift_staying(doc INTEGER PRIMARY KEY)"

    def __init__(self, db: Db, sid: int) -> None:
        self.db, self.sid = db, sid
        rows = db.execute("SELECT path, id, hash FROM docs WHERE source = ? LIMIT ?", (sid, STORED + 1)).fetchall()
        self.few = len(rows) <= STORED
        self.old: dict[str, tuple[int, str]] = {path: (doc, digest) for path, doc, digest in rows} if self.few else {}
        self.given: set[str] = set()  # of the few, the paths the file gave again
        self.first: int | None = None  # the first document added: the file's from it on are this run's
        self.staying = False  # whether any of the many stay

    def at(self, paths: list[str]) -> dict[str, tuple[int, str, bool]]:
        """The file's documents at these paths, by path: (id, hash, whether the file gave the path before in this
        run, so that its document stays as it is or was added)."""
        held = {path: (*self.old[path], path in self.given) for path in paths if path in self.old}
        if self.few and self.first is None:
            return held  # the old ones are all here, and none is added yet: nothing to ask the store
        marked = "id IN (SELECT doc FROM temp.sift_staying)" if self.staying else "0"
        sql = f"SELECT path, id, hash, source, {marked} FROM docs WHERE path IN (SELECT value FROM json_each(?))"
        for path, doc, digest, source, stays in self.db.execute(sql, (json.dumps(paths),)):
            if source == self.sid and path not in held:
                held[path] = (doc, digest, bool(stays) or (self.first is not None and doc >= self.first))
        return held

    def took(self, given: set[str], unchanged: list[int], added: list[int]) -> None:
        """A batch is stored: the paths it gave, the documents (ids) at them that stay as they are, those added."""
        if self.first is None and added:
            self.first = added[0]
        if self.few:
            self.given.update(path for path in given if path in self.old)
        elif unchanged:
            self.db.execute(self.STAYING)
            self.db.executemany("INSERT INTO temp.sift_staying(doc) VALUES(?)", [(doc,) for doc in unchanged])
            self.staying = True

    def drop_the_gone(self) -> int:
        """Delete the documents the file no longer holds: its old ones whose paths it didn't give again. Returns how
        many."""
        if self.few:
            gone = [doc for path, (doc, _) in self.old.items() if path not in self.given]
            store.delete_docs(self.db, gone)
            return len(gone)
        old, args = ("", [self.sid]) if self.first is None else (" AND id < ?", [self.sid, self.first])
        left = " AND id NOT IN (SELECT doc FROM temp.sift_staying)" if self.staying else ""
        sql, dropped = f"SELECT id FROM docs WHERE source = ?{old}{left} LIMIT {DROPPED}", 0
        while gone := [doc for (doc,) in self.db.execute(sql, args)]:
            store.delete_docs(self.db, gone)
            dropped += len(gone)
        if self.staying:
            self.db.execute("DELETE FROM temp.sift_staying")
        return dropped


def _left_out(db: Db, sid: int, rel: str, path: str, note: Note) -> None:
    """Leave out a document of the file `rel` (`sid`) whose path another file's document holds: a record or an
    attachment is a document below its file's name (`data.jsonl` holds `data/a.md`), as a real file may be. The first
    keeps the path; the other waits for it (store.wait), and its file is read again once it is free, not before."""
    sql = "SELECT s.path FROM docs d JOIN sources s ON s.id = d.source WHERE d.path = ?"
    if (held := db.execute(sql, (path,)).fetchone()) is None:  # not the path the store refused
        raise sqlite3.IntegrityError(f"{rel}: {path}: the index refused the document")
    note(f"{path}: a document of {held[0]} has this path; this one of {rel} is left out until it is free")
    store.wait(db, path, sid)


REFRESH_BATCH = 1000  # files read again from the index at a time


def retitle(catalog: Catalog[C], name: str, title_from: str) -> tuple[int, int]:
    """Choose where a collection's Markdown and text files take their titles from (see readers.parse), and retitle
    those indexed from the text its store holds: no file is read again. Returns how many titles changed, and how many
    passages lost their vectors for `sift embed` to embed again."""
    collection = catalog.set_title_from(name, title_from)
    db = catalog.store(collection)
    changed = dropped = 0
    for _, files in _files(db):
        c, d = _reread(db, files, collection.choices)
        changed, dropped = changed + c, dropped + d
        db.commit()  # a batch at a time: other writers wait for no more
    postings.update(db, collection.store, name)
    return changed, dropped


def add_links(db: Db, name: str, fields: Fields) -> None:
    """Read every indexed file's document again from the text the store of the collection `name` holds, as this sift
    reads it with the collection's `fields`, and store its links and names: what an index of sift 1.0 lacks. Front
    matter is read again too, as lists of unquoted items were text before. No file is read; nothing is embedded
    again, unless its title changed (see _reread)."""
    total = db.execute("SELECT count(*) FROM sources").fetchone()[0]
    done = 0
    with console.Progress(total, "files", prefix=f"{name}: links: ") as progress:
        for n, files in _files(db):
            _reread(db, files, fields)
            done += n
            progress.update(done)


def _files(db: Db) -> Iterator[tuple[int, list[FileRow]]]:
    """The documents of the files, REFRESH_BATCH files at a time: how many files, and (id, path, title, meta) of their
    documents. A file's document has the file's path; a JSON Lines file has records instead."""
    sql = """SELECT s.id, d.id, d.path, d.title, d.meta FROM sources s
             LEFT JOIN docs d ON d.path = s.path AND d.source = s.id WHERE s.id > ? ORDER BY s.id LIMIT ?"""
    after = 0
    while rows := db.execute(sql, (after, REFRESH_BATCH)).fetchall():
        after = rows[-1][0]
        yield len(rows), [row[1:] for row in rows if row[1] is not None]


def _reread(db: Db, files: Sequence[FileRow], fields: Fields) -> tuple[int, int]:
    """Read the documents of files again from the index, (id, path, title, meta) of each: Markdown and text files from
    the text the index holds, as this sift reads them with the collection's `fields`, other files only for their
    names. Their names and links are stored anew; titles and front matter that changed go into the documents and the
    keyword index, and passages embedded with a title that changed lose their vectors. A note of front matter alone
    gets its passage (sift 1.0 gave it none). Returns how many documents changed, and how many vectors went."""
    notes = json.dumps([doc for doc, path, _, _ in files if path.lower().endswith(readers.TEXT)])
    texts = dict(db.execute("SELECT id, text FROM docs WHERE id IN (SELECT value FROM json_each(?))", (notes,)))
    sql = "SELECT doc, count(*) FROM chunks WHERE doc IN (SELECT value FROM json_each(?)) GROUP BY doc"
    passages = dict(db.execute(sql, (notes,)))
    changed: dict[int, tuple[str, str, str]] = {}
    fresh: dict[int, list[readers.Chunk]] = {}
    saved: list[links.Saved] = []
    for doc_id, path, title, meta in files:
        if doc_id not in texts:
            saved.append((doc_id, path, json.loads(meta), (), True))
            continue
        doc = readers.note(texts[doc_id], path.rsplit("/", 1)[-1], fields)
        saved.append((doc_id, path, doc.meta, doc.links, True))
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
    sql = "SELECT id, doc, pos, len, seq FROM chunks WHERE doc IN (SELECT value FROM json_each(?)) AND seq > 0"
    later = db.execute(sql, (json.dumps(list(changed)),)).fetchall()
    texts = store.texts(db, ((chunk, doc, pos, length) for chunk, doc, pos, length, _ in later))
    stale = [
        chunk
        for chunk, doc, _, _, seq in later
        if embed.text_of(seq, changed[doc][0], texts[chunk]) != embed.text_of(seq, changed[doc][1], texts[chunk])
    ]
    dropped = store.drop_vectors(db, "chunk IN (SELECT value FROM json_each(?))", (json.dumps(stale),))
    if dropped:
        store.touch(db)
    return len(changed), dropped


class Reindexed(NamedTuple):
    """What `reindex` did: per file, ({"id", "status"[, "note"]}, its documents' row ids); the chunks it removed; and,
    by collection id, what each store it changed said of its vectors before and after: (version before, version after,
    the epoch of its codes after, how many vectors it counted after), read in the change's transaction, for a server to
    follow without loading every code again (see search.Index._patch)."""

    files: list[tuple[Entry, list[int]]]
    removed: list[int]
    versions: dict[int, tuple[str | None, str | None, str | None, int]]


def reindex(catalog: Catalog[C], paths: Sequence[str], scans: bool = True) -> Reindexed:
    """Read files again now, or remove their documents if they are gone, so that the next search finds them as they
    are: their text, front matter, links and keyword entries. `paths` are ids, `collection/path`, or absolute paths in
    a collection, each a file `sift index` would read; if one isn't, nothing is read. A file's status is "added",
    "changed", "unchanged", "removed", "not found", "skipped" (no documents in it, with a "note" saying why) or
    "unreadable". In one transaction per collection. Without `scans`, scans to read with OCR are left for `sift
    index`, which reads their files again even if they don't change."""
    if not _are_paths(paths):
        raise ValueError(f"paths must be a list of ids or absolute paths, got {paths!r}")
    if not paths:
        raise ValueError("paths needs at least one id or absolute path")
    if len(paths) > MOST_FILES:
        raise ValueError(f"at most {MOST_FILES:,} files at a time; `sift index` reads any number")
    files = list(dict.fromkeys(_locate(catalog, path) for path in paths))
    on_disk = [_reindex_read(catalog.store(c), c.path, c.choices, rel, scans) for c, rel in files]
    found: dict[int, tuple[Entry, list[int]]] = {}  # by the file's place in `files`: a collection at a time
    removed: list[int] = []
    versions: dict[int, tuple[str | None, str | None, str | None, int]] = {}
    for collection in dict.fromkeys(c for c, _ in files):
        db = catalog.store(collection)
        db.execute("BEGIN IMMEDIATE")
        try:
            before = store.get_meta(db, "version")
            for i, ((c, rel), found_on_disk) in enumerate(zip(files, on_disk)):
                if c == collection:
                    entry, docs, gone = _reindex_file(db, c.name, rel, *found_on_disk)
                    found[i] = (entry, docs)
                    removed += gone
            versions[collection.id] = (before, *store.vectors_state(db))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        postings.update(db, collection.store, collection.name, rebuild=False)  # quick: what changed, no more
    return Reindexed([found[i] for i in range(len(files))], removed, versions)


def _are_paths(paths: Sequence[object]) -> TypeGuard[list[str] | tuple[str, ...]]:
    """Whether `paths` is a list of paths, as a caller that isn't type-checked may get wrong: one path, not in a list,
    is a sequence of its characters."""
    return isinstance(paths, list | tuple) and all(isinstance(path, str) for path in paths)


def _locate(catalog: Catalog[C], path: str) -> tuple[C, str]:
    """The file an id or absolute path names, as `sift index` stores it: (its collection, its path in it), in the file
    system's spelling. A file `sift index` wouldn't read is an error."""
    collections = sorted(catalog.collections(), key=lambda c: -len(c.path))
    given = os.path.expanduser(path)
    if os.path.isabs(given):
        full = os.path.normpath(given)
        if (held := holding(collections, full)) is None:
            raise ValueError(_nowhere(path, full, collections))
        collection, rel = held
    else:
        name, _, rel = path.replace("\\", "/").partition("/")
        collection, rel = catalog.get(name), posixpath.normpath(rel) if rel else ""
        if rel in ("", "."):
            raise ValueError(f"{path!r}: name a file of the collection, as collection/path, or by its absolute path")
        if rel == ".." or rel.startswith(("../", "/")):
            raise ValueError(f"{path!r} is outside the collection {name!r}")
    name, root = collection.name, collection.path
    # "D:/x.md" is D:'s x.md, "x.md:s" a stream of x.md, ".. /x.md" the parent's x.md: whatever else Windows reads
    if WINDOWS and (other := next((p for p in rel.split("/") if ":" in p or p != p.rstrip(". ")), None)):
        raise ValueError(f"{name}/{rel}: Windows takes {other!r} for another name; give the file's name as it is")
    if not os.path.isdir(root):  # never take an unplugged drive for files deleted
        raise ValueError(f"{name}: folder not found: {root}")
    try:
        parts = _spelled(root, rel.split("/"))
    except ValueError as aliased:
        raise ValueError(f"{name}/{rel}: {aliased}") from None
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
        if _linked(os.path.join(root, *parts[:i])):
            raise ValueError(
                f"{name}/{rel}: {parts[i - 1]}/ is linked from elsewhere, and `sift index` doesn't follow it"
            )
    rel = "/".join(parts)
    try:
        rel.encode()
    except UnicodeEncodeError:  # bytes that aren't UTF-8 (Linux); SQLite stores only valid text
        raise ValueError(f"{name}: unsupported file name {rel!r}") from None
    if not os.path.lexists(os.path.join(root, *parts)):  # gone: the file as indexed, in whatever spelling
        rel = next(iter(_indexed_as(catalog.store(collection), rel)), rel)
    return collection, rel


def inside(root: str, full: str) -> bool:
    """Whether the normalised path `full` lies in the folder `root`, in the file system's case."""
    return os.path.normcase(full).startswith(os.path.normcase(os.path.join(root, "")))


def holding(collections: Sequence[C], full: str) -> tuple[C, str] | None:
    """The collection whose folder holds the normalised absolute path `full`, the nearest, and the path in it, with
    `/`: (collection, rel), or None. As given, else with its folder resolved: a collection keeps its folder as it
    resolves (`sift add`), so a file named through a link to it, a junction, or a drive mapped to a network share is in
    it too. Resolved only then: it asks the file system, which a share that doesn't answer takes long over. The file
    itself stays as named: a linked file is indexed at its link's path."""
    nearest = sorted(collections, key=lambda c: -len(c.path))

    def held(path: str) -> tuple[C, str] | None:
        return next(
            ((c, os.path.relpath(path, c.path).replace(os.sep, "/")) for c in nearest if inside(c.path, path)), None
        )

    return held(full) or held(os.path.join(os.path.realpath(os.path.dirname(full)), os.path.basename(full)))


def _linked(path: str) -> bool:
    """Whether a folder leads elsewhere: a symbolic link, or on Windows a junction, which any user may make."""
    return os.path.islink(path) or _junction(path)


MOUNT_POINT = 0xA0000003  # IO_REPARSE_TAG_MOUNT_POINT, a junction's; module stat names it on Windows alone


def _junction(path: str, entry: os.DirEntry[str] | None = None) -> bool:
    """Whether a folder (its path, or the DirEntry of it) is a Windows junction; nothing elsewhere is."""
    try:
        found = entry.stat(follow_symlinks=False) if entry is not None else os.lstat(path)
    except OSError:
        return False
    return getattr(found, "st_reparse_tag", 0) == MOUNT_POINT


NEAREST_SHOWN = 5  # collections an error about a file in none names, the nearest first


def _nowhere(path: str, full: str, collections: Sequence[C]) -> str:
    """Why the file at `full` is in no collection, and how it could be: the collections' folders, nearest first, and
    the `sift add` that would cover it."""
    parts = Path(os.path.normcase(full)).parts

    def shared(root: str) -> int:  # how many folders, from the drive or root on, the file and the folder have in common
        return len(list(takewhile(lambda pair: pair[0] == pair[1], zip(parts, Path(os.path.normcase(root)).parts))))

    nearest = sorted(collections, key=lambda c: (-shared(c.path), c.name))
    listing = "; ".join(f"{c.name}: {c.path}" for c in nearest[:NEAREST_SHOWN])
    listing += "; ..." if len(nearest) > NEAREST_SHOWN else ""
    head = f"The collections and their folders: {listing}." if collections else "none is registered yet."
    return (
        f"{path} is in no collection{'.' if collections else ':'} {head} To index files there, add their folder as a "
        f"collection: `{console.command('sift', 'add', os.path.dirname(full))}`"
    )


def _spelled(root: str, parts: list[str]) -> list[str]:
    """A path's parts as the file system spells them: on Windows and macOS a path names a file in any case or Unicode
    form, which indexed so would be the file a second time. Parts that don't exist stay as given; one that exists
    under no such name, a Windows short name (PRIVAT~1), is an error: through it, a folder left out was read."""
    folder = root
    spelled: list[str] = []
    for i, part in enumerate(parts):
        try:
            names = os.listdir(folder)
        except OSError:
            return spelled + parts[i:]
        if part not in names:
            same = [n for n in names if readers.fold(n) == readers.fold(part)]
            if len(same) == 1:
                part = same[0]
            elif os.path.lexists(os.path.join(folder, part)):
                raise ValueError(f"{part!r} is another name of a file or folder there; give its name as it is")
        spelled.append(part)
        folder = os.path.join(folder, part)
    return spelled


def _indexed_as(db: Db, rel: str) -> list[str]:
    """The paths of the collection's files that `rel` names, in any case or Unicode form: one, or none."""
    if db.execute("SELECT 1 FROM sources WHERE path = ?", (rel,)).fetchone():
        return [rel]
    paths = [p for (p,) in db.execute("SELECT path FROM sources")]
    same = [p for p in paths if readers.fold(p) == readers.fold(rel)]
    return same if len(same) == 1 else []


def _reindex_read(
    db: Db, root: str, fields: Fields, rel: str, scans: bool
) -> tuple[Info, Loaded | readers.Records, bool]:
    """A file as `reindex` finds it, before anything is written: its stat, the error that kept it from being stat'ed
    (it is there but out of reach, and kept as indexed), or None if it is gone; its documents and problems as `sift
    index` reads them, scans read if `scans`, its Records, unread, if it is read as it is stored, None if it can't be
    read, or the error its reader raised; and whether a scan was left for `sift index`."""
    path = os.path.join(root, *rel.split("/"))
    try:
        info = os.stat(path)  # through a link, as `sift index` reads it
    except GONE:
        info = None
    except OSError as error:
        return error, None, False
    if info is None or not stat.S_ISREG(info.st_mode):
        return None, None, False
    if readers.streamed(path, info.st_size):
        return info, readers.Records(path, fields), False  # read as it is stored
    read = _loaded(path, fields)
    if read is None or isinstance(read, Exception):
        return info, read, False
    docs, problems = read
    docs, left = _read_scans(db, _source(db, rel), rel, docs, problems.append, now=scans)
    return info, (docs, problems), left


def _reindex_file(
    db: Db, name: str, rel: str, info: Info, read: Loaded | readers.Records, left: bool
) -> tuple[Entry, list[int], list[int]]:
    """Bring one file's documents up to date with what `_reindex_read` found, or remove them if it is gone, as `sift
    index` would: ({"id", "status"}, its documents' row ids, the chunks removed). A chunk removed is every chunk of a
    document replaced or removed. A file with a scan `left` is read again by the next `sift index`. One that can't be
    read to its end as it is stored, or whose documents the store refuses (REFUSED), stays as it is indexed, whole,
    "unreadable" with why: the call's other files are read, as `sift index` reads them."""
    entry = {"id": f"{name}/{rel}"}
    sid = _source(db, rel)
    sql = "SELECT c.doc, c.id FROM chunks c JOIN docs d ON d.id = c.doc WHERE d.source = ?"
    chunks = db.execute(sql, (sid,)).fetchall()
    if info is None:
        if sid is None:
            return entry | {"status": "not found"}, [], []
        _drop_source(db, sid)
        return entry | {"status": "removed"}, [], [chunk for _, chunk in chunks]
    if read is None or isinstance(info, OSError):  # an error kept it from being stat'ed: it can't be read either
        return entry | {"status": "unreadable"}, [], []
    if isinstance(read, Exception):  # its reader failed: said, as `sift index` says it, and the others are read
        return entry | {"status": "unreadable", "note": f"{type(read).__name__}: {read}"}, [], []
    problems: list[str]
    docs: Iterable[readers.Doc]
    if isinstance(read, readers.Records):  # read as it is stored, while it is written
        records, docs, problems = read, read, []
    else:
        records, (docs, problems) = None, read
    twice = records.twice if records else None
    db.execute("SAVEPOINT file")
    sid = _source(db, rel, make=True)
    had = db.execute("SELECT count(*) FROM docs WHERE source = ?", (sid,)).fetchone()[0]
    counts: Counts = Counter()
    try:
        _store(db, sid, rel, docs, counts, problems.append, twice)
    except (*REFUSED, readers.Unread) as refused:
        db.execute("ROLLBACK TO file")
        return entry | {"status": "unreadable", "note": str(refused)}, [], []
    finally:
        db.execute("RELEASE file")
    if records and twice:  # read as it was stored: what it skipped is known now
        problems += records.problems
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
    unchanged = set(now)  # a document replaced has a new id: ids are never handed out twice
    return entry | {"status": status} | note, now, [chunk for doc, chunk in chunks if doc not in unchanged]


@overload
def _source(db: Db, rel: str, make: Literal[True]) -> int: ...


@overload
def _source(db: Db, rel: str, make: bool = False) -> int | None: ...


def _source(db: Db, rel: str, make: bool = False) -> int | None:
    """The id of the file `rel` as the store holds it now, made if `make`; None if there is none. Not as scanned: a
    server's `reindex` may have added the file or dropped it since."""
    if make:
        db.execute("INSERT OR IGNORE INTO sources(path) VALUES(?)", (rel,))
    row = db.execute("SELECT id FROM sources WHERE path = ?", (rel,)).fetchone()
    return row and row[0]


def _drop_source(db: Db, sid: int) -> int:
    docs = [d for (d,) in db.execute("SELECT id FROM docs WHERE source=?", (sid,))]
    store.stop_waiting(db, sid)
    store.delete_docs(db, docs)
    db.execute("DELETE FROM sources WHERE id=?", (sid,))
    return len(docs)


def walk(root: str, ignore: Ignore | None = None, unlisted: Unlisted | None = None) -> Iterator[tuple[str, int, int]]:
    """Yield (path, mtime_ns, size) for the files sift has a reader for, but those in hidden folders, SKIP_DIRS and
    what `ignore`, the collection's .siftignore, leaves out. DirEntry.stat() is free on Windows. What is there but out
    of reach for the moment (access denied, a share that doesn't answer) goes into `unlisted`, if given, as (its path
    relative to root, the error): a folder that can't be listed, or whose listing breaks off, with a slash, or "" for
    root; a file that can't be stat'ed. What is gone meanwhile, a broken link too, is passed over."""

    def unreachable(rel: str, error: OSError) -> None:
        if unlisted is not None:
            unlisted.append((rel, error))

    stack: list[tuple[str, str]] = [(root, "")]
    while stack:
        folder, rel = stack.pop()
        try:
            entries = os.scandir(folder)
        except GONE:
            continue
        except OSError as error:
            unreachable(rel, error)
            continue
        with entries:
            try:
                for e in entries:
                    if e.is_dir(follow_symlinks=False):
                        if (
                            not e.name.startswith(".")
                            and e.name not in SKIP_DIRS
                            and not (ignore and ignore(rel + e.name, True))
                            and not _junction(e.path, e)
                        ):
                            stack.append((e.path, rel + e.name + "/"))
                    elif readers.extension(e.name) in readers.READERS and not (ignore and ignore(rel + e.name)):
                        try:
                            st = e.stat()
                        except GONE:  # e.g. a broken symlink
                            continue
                        except OSError as error:
                            unreachable(rel + e.name, error)
                            continue
                        if stat.S_ISREG(st.st_mode):  # no pipe, which has no end to read to, nor a folder so named
                            yield e.path, st.st_mtime_ns, st.st_size
            except OSError as error:  # the listing broke off
                unreachable(rel, error)


def index(catalog: Catalog[C], names: Sequence[str] | None = None) -> None:
    """Bring the stores of the collections named, or of every active one, in sync with the files on disk. Only new or
    changed files are read, and of those only new or changed documents are indexed again."""
    for name in names or ():
        catalog.get(name)  # an unknown name is an error that names those there are, before anything is read
    for c in catalog.collections():
        if names and c.name not in names:
            continue
        if not (names or c.active):
            console.say(f"{c.name}: inactive, skipped")
            continue
        if not os.path.isdir(c.path):  # never treat an unplugged drive as "all files deleted"
            console.say(f"{c.name}: folder not found, skipped: {c.path}")
            continue
        try:
            ignore = Ignore.of(c.path)
        except OSError as error:  # indexing all would take in the very files it keeps out
            console.say(f"{c.name}: skipped, as its {SIFTIGNORE} can't be read: {error}")
            continue
        try:
            db = catalog.opened(c)
        except LookupError as missing:
            console.say(f"{missing}; skipped")
            continue
        try:
            _index_collection(db, c.name, c.path, c.choices, ignore)
            store.list_basenames(db, c.name)
            postings.update(db, c.store, c.name, cpus() - 1)
        except sqlite3.DatabaseError as error:  # one store damaged stops no other collection
            db.rollback()
            console.say(f"{store.damage(c.name, error)}; skipped")


def _index_collection(db: Db, name: str, root: str, fields: Fields, ignore: Ignore) -> None:
    """Bring one collection in sync: drop what is gone, read what is new or changed; its keyword postings follow
    after (see `postings`)."""
    t0 = time.perf_counter()
    _read_skipped_scans(db)
    store.read_ansi_again(db, name)
    known: dict[str, tuple[int, int, int]] = {
        p: (i, m, s) for i, p, m, s in db.execute("SELECT id, path, mtime, size FROM sources")
    }
    console.say(f"{name}: scanning {root}")  # a large or cloud folder takes a while
    todo, seen, unlisted = _scan(root, known, ignore)
    kept = _kept(name, known.keys() - seen, unlisted)
    counts: Counts = Counter()
    for rel in _gone(name, root, known.keys() - seen - kept, seen):
        counts["removed"] += _drop_source(db, known[rel][0])
    db.commit()  # before any file is read: other writers wait for none of it
    todo += _freed(db, root, seen - {item[1] for item in todo})  # those that waited for a path these drops freed
    while todo:
        db.commit()  # what `_freed` wrote: no lock held while the files are read
        with console.Progress(len(todo), "files", prefix=f"{name}: ") as progress:
            _read_and_store(db, name, fields, todo, counts, progress)
        todo = _freed(db, root, seen)  # and for one freed as those were stored: each freed once, so this ends
    db.commit()
    unreadable = f", {counts['unreadable']} unreadable" if counts["unreadable"] else ""
    console.say(
        f"{name}: {len(seen)} files, {counts['added']} added, {counts['changed']} changed, {counts['removed']} removed"
        f"{unreadable} ({time.perf_counter() - t0:.1f}s)"
    )


def _gone(name: str, root: str, unseen: set[str], seen: set[str]) -> set[str]:
    """The files not seen whose documents go: all of them, unless no file at all was seen. A folder left without
    any of its files is one emptied, but also a drive not mounted, whose folder stays, and a folder being filled anew:
    its documents, and the vectors embedded for them, are kept, and said, with how to let them go."""
    if seen or not unseen:
        return unseen
    console.say(
        f"{name}: none of its {console.count(len(unseen), 'file')} is left in {root}: kept as indexed, as a drive "
        f"not mounted and a folder being filled anew look so. If they are gone for good, "
        f"`{console.command('sift', 'rm', name)}` forgets them"
    )
    return set()


def _freed(db: Db, root: str, among: set[str]) -> list[Item]:
    """The files to read now, of those `among`, whose documents waited for a path freed meanwhile (see store.wait)."""
    freed: list[Item] = []
    for rel in [rel for rel in store.freed(db) if rel in among]:
        path = os.path.join(root, *rel.split("/"))
        with contextlib.suppress(OSError):  # gone meanwhile: the next index drops it
            info = os.stat(path)
            freed.append((path, rel, info.st_mtime_ns, info.st_size))
    return freed


def _kept(name: str, unseen: set[str], unlisted: Unlisted) -> set[str]:
    """The files not seen that are kept as indexed, as they lie in a folder that couldn't be listed: an unanswering
    share or a denied folder is no folder deleted. Says so for each folder."""
    kept: set[str] = set()
    for where, error in unlisted:
        why = f"{name}: {where or 'its folder'} can't be read ({error.strerror or error})"
        if where and not where.endswith("/"):  # a file
            kept |= {where} & unseen
            console.say(f"{why}: {'kept as indexed' if where in unseen else 'left for the next index'}")
            continue
        inside = {rel for rel in unseen if rel.startswith(where)}
        kept |= inside
        console.say(f"{why}: {console.count(len(inside), 'file')} in it kept as indexed")
    return kept


def _scan(root: str, known: dict[str, tuple[int, int, int]], ignore: Ignore) -> tuple[list[Item], set[str], Unlisted]:
    """The files to read, new or changed since `known` (path -> id, mtime, size), every file seen but those `ignore`
    leaves out, and the folders that couldn't be listed (see `walk`)."""
    todo: list[Item] = []
    seen: set[str] = set()
    unlisted: Unlisted = []
    for path, mtime, size in walk(root, ignore, unlisted):
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
    return todo, seen, unlisted


def _read_and_store(
    db: Db, name: str, fields: Fields, todo: list[Item], counts: Counts, progress: console.Progress
) -> None:
    """Read the files to do, in worker processes when it pays off, and store their documents batch by batch."""
    batches, whole = _batches(todo), _whole(todo)
    n = workers(len(whole), sum(item[3] for item in whole))  # parsing is CPU-bound: processes, not threads
    with _Workers(n, fields) if n > 1 else nullcontext() as pool:
        # double buffering: the next batch is parsed while the current one is written
        pending = _read(pool, fields, batches[0])
        done, step = 0, max(1, len(todo) // PROGRESS_LINES)  # an update per file, or fewer for many small ones
        for b, batch in enumerate(batches):
            loaded = pending
            if b + 1 < len(batches):
                pending = _read(pool, fields, batches[b + 1])
            results = loaded()  # read before the batch's first write: no lock held while files are read
            for (path, rel, mtime, size), res in zip(batch, results):
                done += 1
                if done % step == 0 or done == len(todo):
                    progress.update(done)
                note: Note = lambda text: console.say(f"{name}: {text}")  # noqa: E731
                if isinstance(res, readers.Records):  # read as it is stored: what it skipped is known after
                    if _stored(db, rel, res, (mtime, size), counts, note, res.twice):
                        for problem in res.problems:
                            note(f"{rel}: {problem}")
                    continue
                if res is None:  # unreadable now; retried on the next run
                    console.say(f"unreadable, skipped: {path}")
                    counts["unreadable"] += 1
                    continue
                if isinstance(res, Exception):  # a reader failed: said, and read again by the next run
                    note(f"{rel}: skipped ({type(res).__name__}: {res})")
                    counts["unreadable"] += 1
                    continue
                docs, problems = res
                for problem in problems:
                    note(f"{rel}: {problem}")
                docs, left = _read_scans(db, _source(db, rel), rel, docs, note)
                _stored(db, rel, docs, (None if left else mtime, size), counts, note)
            db.commit()


REFUSED = (ValueError, RecursionError, sqlite3.DataError, sqlite3.IntegrityError)  # Recursion: Python 3.11's JSON


def _stored(
    db: Db,
    rel: str,
    docs: Iterable[readers.Doc],
    stat: tuple[int | None, int],
    counts: Counts,
    note: Note,
    twice: Twice = None,
) -> bool:
    """Store a file's documents and its (mtime, size), in a savepoint of the batch's transaction: if the store refuses
    them, as text SQLite can't hold, or the file can't be read to its end as it is stored, only this file is left
    out, whole, said, and read again by the next run. Returns whether it is stored."""
    if not db.in_transaction:
        db.execute("BEGIN")  # so that releasing the savepoint commits nothing
    db.execute("SAVEPOINT file")
    before, stored = counts.copy(), False
    try:
        sid = _source(db, rel, make=True)
        _store(db, sid, rel, docs, counts, note, twice)
        db.execute("UPDATE sources SET mtime=?, size=? WHERE id=?", (*stat, sid))
        stored = True
    except (*REFUSED, readers.Unread) as refused:
        db.execute("ROLLBACK TO file")
        counts.clear()
        counts.update(before)  # Counter.update adds: onto nothing, as they were
        counts["unreadable"] += 1
        note(f"{rel}: skipped ({refused})")
    db.execute("RELEASE file")
    return stored
