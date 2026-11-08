"""`sift embed`: embedding every chunk that has no vector yet, collection by collection and slab by slab, with progress.
The models come from outside (`models.default()` unless given), so this runs the same on a GPU, a CPU or through a
worker.

The models run on the main thread, as PyTorch needs, and never wait for the store: a thread reads the slabs ahead and
sorts them by length, another writes their vectors, both with the store's one connection, in turn."""

import collections
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import console, store
from . import models as _models

SLAB = 20000  # chunks per commit; length sorting happens within a slab
AHEAD = 2  # slabs read ahead of the model


def text_of(seq, title, text):
    """What gets embedded: chunks after the first repeat their document title, unless they already start with it."""
    return text if seq == 0 or text.lstrip("# ").startswith(title) else f"{title}\n\n{text}"


PENDING = "chunks c JOIN docs d ON d.id = c.doc WHERE NOT EXISTS (SELECT 1 FROM codes k WHERE k.chunk = c.id)"
"""The chunks still without a vector, with their documents, as a FROM clause: those without a code, which every
vector of the store's length has. Looked up among the vectors, each lookup read a page of vectors: 37 GB for 25
million PubMed chunks, against 2.5 GB of codes."""


def _pending(db):
    """How many chunks have no vector yet, from the counts the store keeps: counting them read every chunk and its
    vector, 9 minutes over 30 million from a cold disk. A chunk without a document counts; the slabs pass it over."""
    counted = store.counts(db)
    return max(counted["chunks"] - counted["vectors"], 0)


def embed_pending(catalog, reset=False, models=None, collections=None):
    """Embed every chunk without a vector, of the `collections` named or of every active one, slab by slab, with
    `models` (default: `models.default()`). Safe to interrupt and resume."""
    if reset and collections:
        raise SystemExit(
            "--reset embeds every collection again, since all must share one model: "
            "run `sift embed --reset` without collection names"
        )
    chosen = _chosen(catalog, collections)
    models = models or _models.default()
    if reset:
        for db in [catalog.store(c) for c in catalog.collections()]:  # every one opens, or none changes
            _forget_vectors(db)
    using = models.name
    stores = list(_stores(catalog, chosen, named=bool(collections)))
    for c, db in stores:
        _check_model(db, c.name, using)
        _drop_broken_vectors(db)
    pending = [(c, db, _pending(db)) for c, db in stores]
    total, named = sum(n for *_, n in pending), ", ".join(collections or ())
    if not total:
        console.say(f"nothing to embed in {named}" if named else "nothing to embed")
        return
    of = f" of {named}" if named else ""
    console.say(f"embedding {console.count(total, 'chunk')}{of} with {using} on {models.address}")
    models.prepare()
    with console.Progress(total, "chunks") as progress:
        done, broken = _embed(models, [db for _, db, n in pending if n], using, progress)
    console.say(f"embedded {done - broken:,} chunks in {console.duration(progress.elapsed)}")
    if broken:
        console.say(f"{console.count(broken, 'chunk')} could not be embedded (the model returned NaN); try again")


def _embed(models, stores, using, progress):
    """Embed the slabs of these stores as the reader brings them, and hand each to the writer once all its vectors
    are in. Returns (chunks done, chunks the model broke)."""
    turn = threading.Lock()  # the store's connection, which the reader and the writer take in turn
    on_gpu, done = collections.deque(), 0  # (slab, part) of each batch handed to the model, not yet back
    with _Reader(models, stores, turn) as slabs, _Writer(turn, using) as writer:

        def batches():
            for slab in slabs:
                for part in slab.parts:
                    on_gpu.append((slab, part))
                    yield [slab.texts[i] for i in part]

        for vectors in models.encode_batches(batches()):
            slab, part = on_gpu.popleft()
            if slab.add(part, vectors):
                writer.write(slab)
            done += len(part)
            progress.update(done)
    return done, writer.broken


class _Slab:
    """Chunks embedded and written together: their ids and texts, in batches of similar length (parts: indexes into
    texts), and their vectors as they come."""

    def __init__(self, db, ids, texts, parts):
        self.db, self.ids, self.texts, self.parts = db, ids, texts, parts
        self.vectors, self._left = None, len(texts)

    def add(self, part, vectors):
        """Take the vectors of a part; returns whether all are in."""
        if self.vectors is None:
            self.vectors = np.empty((len(self.texts), vectors.shape[1]), np.float16)
        self.vectors[part] = vectors
        self._left -= len(part)
        return not self._left


_DONE = object()


class _Reader:
    """A thread that reads the slabs still to embed, AHEAD at most, sorted by length; iterated, the slabs as they come.
    An error in the thread is raised where they're taken."""

    def __init__(self, models, stores, turn):
        self.models, self.stores, self.turn = models, stores, turn
        self.slabs, self.stop = queue.Queue(AHEAD), threading.Event()
        self.thread = threading.Thread(target=self._read, name="sift-embed-read", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def __iter__(self):
        while (slab := self.slabs.get()) is not _DONE:
            if isinstance(slab, BaseException):
                raise slab
            yield slab

    def _read(self):
        try:
            for db in self.stores:
                last = -1
                while not self.stop.is_set() and (rows := self._rows(db, last)):
                    texts = [text_of(*row[1:]) for row in rows]
                    self._put(_Slab(db, [row[0] for row in rows], texts, by_length(self.models, texts)))
                    last = rows[-1][0]
            self._put(_DONE)
        except BaseException as error:
            self._put(error)

    def _rows(self, db, last):
        with self.turn:
            return db.execute(SLAB_SQL, (last, SLAB)).fetchall()

    def _put(self, item):
        """Queue an item, unless embedding stops meanwhile."""
        while not self.stop.is_set():
            try:
                self.slabs.put(item, timeout=0.1)
                return
            except queue.Full:
                continue


SLAB_SQL = f"""SELECT c.id, c.seq, d.title, substr(d.text, c.pos + 1, c.len) FROM {PENDING} AND c.id > ?
               ORDER BY c.id LIMIT ?"""


class _Writer:
    """A thread that writes the vectors of each slab handed to it, in order, and adds up the chunks the model broke.
    An error in it is raised at the next slab handed over, or at the end."""

    def __init__(self, turn, model):
        self.turn, self.model, self.broken, self._writes = turn, model, 0, []
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="sift-embed-write")

    def __enter__(self):
        return self

    def __exit__(self, kind, *exc):
        self._pool.shutdown(wait=True)  # what was embedded is written, even when embedding stops
        if kind is None:
            self._settle(self._writes)

    def write(self, slab):
        self._settle([w for w in self._writes if w.done()])
        self._writes.append(self._pool.submit(self._store, slab))

    def _store(self, slab):
        with self.turn:
            try:
                return _store_vectors(slab.db, slab.ids, slab.vectors, self.model)
            except BaseException:
                slab.db.rollback()  # none of the slab, rather than part of it in the next slab's commit
                raise

    def _settle(self, writes):
        for write in list(writes):
            self.broken += write.result()
            self._writes.remove(write)


def _chosen(catalog, names):
    """The collections named, or every active one. An unknown name is an error that names those there are."""
    if names:
        return [catalog.get(name) for name in dict.fromkeys(names)]
    return [c for c in catalog.collections() if c.active]


def _stores(catalog, collections, named):
    """The collections with their stores: one whose store is missing is passed over, as `sift index` does, unless
    `named`, which makes it an error."""
    for c in collections:
        try:
            yield c, catalog.store(c)
        except LookupError as missing:
            if named:
                raise
            console.say(f"{missing}; skipped")


def _forget_vectors(db):
    store.drop_vectors(db, "1")
    db.execute("DELETE FROM meta WHERE key IN ('model', 'dim', 'kw_ordered')")  # kw_ordered: see postings._unordered
    store.touch(db)
    db.commit()


def _check_model(db, name, using):
    """A collection embedded already must have been embedded with the model in use."""
    model = store.get_meta(db, "model")
    if model and model != using:
        raise SystemExit(f"{name} was embedded with {model}; run `sift embed --reset` to switch to {using}")


def _drop_broken_vectors(db):
    """sift 0.5 and older embedded in fp16, where EmbeddingGemma overflows now and then into NaN vectors. Drop those,
    once per store, so they are embedded again."""
    if store.get_meta(db, "vectors_checked"):
        return
    if n := store.counts(db)["vectors"]:
        console.say(f"checking {n:,} vectors for broken ones (once) ...")
        broken, cursor = [], db.execute("SELECT chunk, vec FROM vectors")
        while rows := cursor.fetchmany(100_000):
            vectors = np.frombuffer(b"".join(vec for _, vec in rows), np.float16).reshape(len(rows), -1)
            broken += [rows[i][0] for i in np.flatnonzero(~np.isfinite(vectors).all(axis=1))]
        if broken:
            store.drop_vectors(db, "chunk IN (SELECT value FROM json_each(?))", (json.dumps(broken),))
            store.touch(db)
            console.say(f"{console.count(len(broken), 'broken vector')} dropped, to be embedded again")
    store.set_meta(db, "vectors_checked", 1)
    db.commit()


def by_length(models, texts):
    """The texts in batches of similar length to the model, longest first, as lists of indexes into them: each batch
    is padded to its longest text, and characters tell tokens poorly."""
    lengths, step = models.lengths(texts), models.batch
    order = sorted(range(len(texts)), key=lambda i: -lengths[i])
    return [order[start : start + step] for start in range(0, len(order), step)]


def encode(models, texts):
    """Vectors for texts, embedded in batches of similar length (by_length), one call to the models each."""
    vectors = None
    for part in by_length(models, texts):
        out = models.encode([texts[i] for i in part])
        if vectors is None:
            vectors = np.empty((len(texts), out.shape[1]), np.float16)
        vectors[part] = out
    return vectors


def _store_vectors(db, chunks, vectors, model):
    """Store the intact vectors of the chunks still there, and commit. A NaN vector, an overflow inside the model,
    stays out, so its chunk stays pending; so does the vector of a chunk another sift deleted meanwhile, which would
    count as the vector of the chunk that took its place. Returns how many the model broke."""
    intact = np.isfinite(vectors).all(axis=1)
    db.execute("BEGIN IMMEDIATE")  # what is still there, as it is written
    sql = "SELECT id FROM chunks WHERE id IN (SELECT value FROM json_each(?))"
    there = {c for (c,) in db.execute(sql, (json.dumps([int(c) for c in chunks]),))}
    keep = intact & np.fromiter((c in there for c in chunks), bool, len(chunks))
    store.put_vectors(db, [chunk for chunk, ok in zip(chunks, keep) if ok], vectors[keep])
    store.set_meta(db, "model", model)
    store.set_meta(db, "dim", vectors.shape[1])
    store.touch(db)  # tells a running server the codes changed
    db.commit()
    return int((~intact).sum())
