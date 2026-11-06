"""Splitting an index of sift 1.7 or older, one database holding every collection, into a store per collection (see
`store`) and the catalog (see `catalog`): once, on first use, and nothing is read from the files or embedded again.

The old database is read table by table in the order of its rows, each row going to its collection's store with its
ids moved to the collection's (store.first_id): one pass over a large index, where a pass per collection would read
all of it once per collection. Each store then gets its keyword index, made anew, and what an older layout lacks: the
links of sift 1.0's (layout 3), the codes of 1.1 to 1.6's (layout 4). A store is complete once it holds every row
sent to it; stopped before, it is made again. Only when every store is complete does the catalog replace the old
database, which is held throughout, so that no other sift changes it meanwhile.
"""

import os
import shutil
import sqlite3
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import console, store

BATCH = 50_000  # rows read at a time
PARALLEL_CHUNKS = 1_000_000  # with more chunks than this, the stores are finished in processes of their own
INDEXES = ("docs_source", "chunks_doc", "links_doc", "links_key", "names_doc", "names_key")  # made after the rows
CARRIED = ("model", "dim", "vectors_checked", *store.READ_AGAIN)  # what the old meta says of every store
GB = 1e9


def split(path, into):
    """Split the index at `path` into a store per collection in the folder `into`, and put the catalog in its place.
    Interrupted, it goes on where it stopped when run again."""
    into = Path(into).expanduser().resolve()
    with store.alone(path, "the index") as held:
        layout = held.execute("PRAGMA user_version").fetchone()[0]
        collections = held.execute("SELECT id, name, path, fields FROM collections ORDER BY id").fetchall()
        meta = dict(held.execute("SELECT key, value FROM meta"))
        files = {cid: into / f"{name}.sqlite" for cid, name, _, _ in collections}
        pending = {cid: file for cid, file in files.items() if not _complete(file, cid)}
        if pending:
            _space(path, into)
            console.say(
                f"sift 1.8 keeps each collection's index in a file of its own: splitting {path} "
                f"({os.path.getsize(path) / GB:.1f} GB) into {into}, once ..."
            )
            start = time.perf_counter()
            parts = {cid: _begin(Path(f"{file}.part"), cid, meta) for cid, file in pending.items()}
            try:
                sent = _copy(held, parts, layout)
            finally:
                for part in parts.values():
                    part.close()
            names = {cid: (name, fields) for cid, name, _, fields in collections}
            jobs = [(f"{file}.part", str(file), cid, *names[cid], layout, sent[cid]) for cid, file in pending.items()]
            _finish_all(jobs, sum(sent[cid]["chunks"] for cid in pending))
            console.say(f"split in {console.duration(time.perf_counter() - start)}")
        catalog = _catalog(Path(f"{path}.next"), collections, files)
    _replace(path, catalog)


def _complete(file, cid):
    """Whether `file` is the complete store of the collection `cid`, from a split stopped later."""
    if not file.exists():
        return False
    store.connect(file, cid).close()  # an error if it is another collection's, or no store
    return True


def _space(path, into):
    """An error unless `into` has room for the stores: as much as the old index takes, at most."""
    into.mkdir(parents=True, exist_ok=True)
    need, free = os.path.getsize(path), shutil.disk_usage(into).free
    if free < need:
        raise SystemExit(
            f"splitting the index takes up to {need / GB:.1f} GB in {into}, which has {free / GB:.1f} GB free: free "
            f"space there, or split it into another folder: `{console.command('sift', 'move', 'FOLDER')}`"
        )


def _begin(part, cid, meta):
    """A new store at `part`, written without a journal: until it is complete, it is made again rather than repaired.
    Its secondary indexes come after its rows, which is faster than keeping them up to date."""
    for file in (part, Path(f"{part}-journal")):
        file.unlink(missing_ok=True)
    db = sqlite3.connect(part)
    db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-524288;")
    store.make(db, cid, read=False)
    for key in CARRIED:
        if key in meta:
            store.set_meta(db, key, meta[key])
    store.touch(db)
    db.executescript("".join(f"DROP INDEX {index};" for index in INDEXES))
    db.commit()
    return db


def _copy(held, parts, layout):
    """Send every row of the old index to its collection's store, ids moved; returns how many rows of each table each
    store got. Rows of collections not pending, and chunks and vectors whose document is gone, are passed over."""
    import numpy as np

    sent = {cid: Counter() for cid in parts}
    first = {cid: store.first_id(cid) for cid in parts}
    most = held.execute("SELECT max(id) FROM docs").fetchone()[0] or 0
    doc_owner = np.full(most + 1, -1, np.int64)  # the collection of each document, by its old id
    most = held.execute("SELECT max(id) FROM chunks").fetchone()[0] or 0
    chunk_owner = np.full(most + 1, -1, np.int64)

    def owners(ids, of):
        ids = np.asarray(ids, np.int64)
        found = np.full(len(ids), -1, np.int64)
        inside = ids < len(of)
        found[inside] = of[ids[inside]]
        return found.tolist()

    def table(name, select, insert, rows_of):
        """Copy a table: rows_of(rows) gives the collection of each row and the row as its store takes it."""
        total = held.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        cursor, done = held.execute(select), 0
        with console.Progress(total, "rows", prefix=f"{name}: ") as progress:
            while rows := cursor.fetchmany(BATCH):
                batches = {}
                for cid, row in rows_of(rows):
                    if cid in parts:
                        batches.setdefault(cid, []).append(row)
                for cid, batch in batches.items():
                    parts[cid].executemany(insert, batch)
                    sent[cid][name] += len(batch)
                done += len(rows)
                progress.update(done)
        for part in parts.values():
            part.commit()

    table(
        "sources",
        "SELECT id, collection, path, mtime, size FROM sources ORDER BY id",
        "INSERT INTO sources(id, path, mtime, size) VALUES(?, ?, ?, ?)",
        lambda rows: [(row[1], (row[0], *row[2:])) for row in rows],
    )

    def docs(rows):
        doc_owner[[row[0] for row in rows]] = [row[1] for row in rows]
        return [(row[1], (row[0] + first.get(row[1], 0), *row[2:])) for row in rows]

    table(
        "docs",
        "SELECT id, collection, source, path, hash, title, meta, text FROM docs ORDER BY id",
        "INSERT INTO docs(id, source, path, hash, title, meta, text) VALUES(?, ?, ?, ?, ?, ?, ?)",
        docs,
    )

    def chunks(rows):
        cids = owners([row[1] for row in rows], doc_owner)
        chunk_owner[[row[0] for row in rows]] = cids
        shift = [first.get(cid, 0) for cid in cids]
        return [(cid, (row[0] + s, row[1] + s, *row[2:])) for cid, s, row in zip(cids, shift, rows)]

    table(
        "chunks",
        "SELECT id, doc, seq, line_start, line_end, pos, len FROM chunks ORDER BY id",
        "INSERT INTO chunks(id, doc, seq, line_start, line_end, pos, len) VALUES(?, ?, ?, ?, ?, ?, ?)",
        chunks,
    )

    def by_chunk(rows):
        cids = owners([row[0] for row in rows], chunk_owner)
        return [(cid, (row[0] + first.get(cid, 0), row[1])) for cid, row in zip(cids, rows)]

    table("vectors", "SELECT chunk, vec FROM vectors ORDER BY chunk", "INSERT INTO vectors VALUES(?, ?)", by_chunk)
    if layout >= 5:
        table("codes", "SELECT chunk, bits FROM codes ORDER BY chunk", "INSERT INTO codes VALUES(?, ?)", by_chunk)
    if layout >= 4:
        table(
            "links",
            "SELECT collection, doc, field, line, kind, target, key, fragment, shown FROM links ORDER BY id",
            "INSERT INTO links(doc, field, line, kind, target, key, fragment, shown) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            lambda rows: [(row[0], (row[1] + first.get(row[0], 0), *row[2:])) for row in rows],
        )
        table(
            "names",
            "SELECT collection, doc, key, kind FROM names ORDER BY rowid",
            "INSERT INTO names(doc, key, kind) VALUES(?, ?, ?)",
            lambda rows: [(row[0], (row[1] + first.get(row[0], 0), *row[2:])) for row in rows],
        )
    return sent


def _finish_all(jobs, chunks):
    """Finish the stores: in processes of their own when there are many chunks, as the keyword index of each takes
    a while; else here."""
    console.say(f"making the keyword index of {console.count(len(jobs), 'collection')} ...")
    if chunks > PARALLEL_CHUNKS and len(jobs) > 1:
        import multiprocessing

        from .indexing import _cpus

        spawn = multiprocessing.get_context("spawn")  # as indexing: forking a process with threads can deadlock
        with ProcessPoolExecutor(min(len(jobs), _cpus()), mp_context=spawn) as pool:
            for said in pool.map(_finish, *zip(*jobs)):
                console.say(said)
    else:
        for job in jobs:
            console.say(_finish(*job))


def _finish(part, file, cid, name, fields, layout, sent):
    """Complete the store at `part`: its secondary indexes, its keyword index, and the links and codes an older layout
    lacks; then give it its name `file`. An error if it doesn't hold every row `sent` to it."""
    import json

    from . import indexing

    start = time.perf_counter()
    db = sqlite3.connect(part)
    try:
        db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-524288;")
        db.executescript(store.SCHEMA + store.LINKS)  # the secondary indexes dropped for the copy
        db.execute("INSERT INTO fts(rowid, title, text, meta) SELECT id, title, text, meta FROM chunk_text")
        db.commit()
        for table, n in sent.items():
            held = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            if held != n:
                raise SystemExit(f"{name}: its new index holds {held:,} {table}, not {n:,}: the split stopped")
        if layout < 4:  # reading the notes again may give a note a passage, and take a retitled one's vector
            indexing.add_links(db, name, json.loads(fields) if fields else {})
            db.commit()
        if layout < 5:
            store.add_codes(db)
        db.execute("PRAGMA journal_mode=WAL")
    finally:
        db.close()
    os.replace(part, file)
    return f"{name}: done in {console.duration(time.perf_counter() - start)}"


def _catalog(path, collections, files):
    """The catalog of these collections, their stores `files`, made at `path` and closed."""
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    try:
        from .catalog import SCHEMA, SCHEMA_VERSION

        db.executescript(SCHEMA + f"PRAGMA user_version = {SCHEMA_VERSION};")
        db.executemany(
            "INSERT INTO collections(id, name, path, fields, store) VALUES(?, ?, ?, ?, ?)",
            [(cid, name, folder, fields, str(files[cid])) for cid, name, folder, fields in collections],
        )
        db.commit()
    finally:
        db.close()
    return path


def _replace(path, catalog):
    """Put the catalog in the old index's place. Its log must be gone: SQLite would apply it to the catalog."""
    log = Path(f"{path}-wal")
    if log.exists() and log.stat().st_size:
        raise SystemExit(f"another sift has {path} open: stop it, then start sift again to finish the split")
    for file in (log, Path(f"{path}-shm")):
        file.unlink(missing_ok=True)
    try:
        os.replace(catalog, path)
    except PermissionError:
        raise SystemExit(f"another sift has {path} open: stop it, then start sift again to finish the split") from None
    console.say(f"the index is split; {path} lists the collections now, and their indexes lie beside each other")
