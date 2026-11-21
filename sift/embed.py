"""`sift embed`: embedding every chunk that has no vector yet, collection by collection and slab by slab, with progress.
The models come from outside (`models.default()` unless given), so this runs the same on a GPU, a CPU or through a
worker.

The models run on the main thread, as PyTorch needs, and never wait for the store: a thread reads the slabs ahead and
sorts them by length, another writes their vectors, both with the store's one connection, in turn."""

import collections
import json
import queue
import sqlite3
import threading
from collections.abc import Generator, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import Self, TypeVar

import numpy as np
from numpy.typing import NDArray

from . import console, store
from . import models as _models

C = TypeVar("C", bound=store.Collection)

SLAB = 20000  # chunks per commit; length sorting happens within a slab
AHEAD = 2  # slabs read ahead of the model

Passage = tuple[int, int, str, str]  # (chunk id, seq, title, text)
Vectors = NDArray[np.float16]


def text_of(seq: int, title: str, text: str) -> str:
    """What gets embedded: chunks after the first repeat their document title, unless they already start with it."""
    return text if seq == 0 or text.lstrip("# ").startswith(title) else f"{title}\n\n{text}"


PENDING = "chunks c JOIN docs d ON d.id = c.doc WHERE NOT EXISTS (SELECT 1 FROM codes k WHERE k.chunk = c.id)"
"""The chunks still without a vector, with their documents, as a FROM clause: those without a code, which every
vector of the store's length has. Looked up among the vectors, each lookup read a page of vectors: 37 GB for 25
million PubMed chunks, against 2.5 GB of codes."""


def passages(db: sqlite3.Connection, chosen: str, args: Sequence[object] = ()) -> list[Passage]:
    """What is embedded of the chunks that `chosen` selects, a FROM clause over chunks `c` and their documents `d`,
    with `args`: (id, seq, title, text) of each, as text_of takes the last three, in its order."""
    rows = db.execute(f"SELECT c.id, c.seq, d.title, c.doc, c.pos, c.len FROM {chosen}", args).fetchall()
    texts = store.texts(db, (row[:1] + row[3:] for row in rows))
    return [(chunk, seq, title, texts[chunk]) for chunk, seq, title, *_ in rows if chunk in texts]


def _pending(db: sqlite3.Connection) -> int:
    """How many chunks have no vector yet, from the counts the store keeps: counting them read every chunk and its
    vector, 9 minutes over 30 million from a cold disk. A chunk without a document counts; the slabs pass it over."""
    counted = store.counts(db)
    return max(counted["chunks"] - counted["vectors"], 0)


def embed_pending(
    catalog: store.Catalog[C],
    reset: bool = False,
    models: _models.Models | None = None,
    collections: Sequence[str] | None = None,
) -> None:
    """Embed every chunk without a vector, of the `collections` named or of every active one, slab by slab, with
    `models` (default: `models.default()`). Safe to interrupt and resume."""
    if reset and collections:
        raise SystemExit(
            "--reset embeds every collection again, since all must share one model: "
            "run `sift embed --reset` without collection names"
        )
    chosen = _chosen(catalog, collections)
    models = models or _models.default()
    using = models.name  # a worker's: asked of it
    if reset:
        models.prepare()  # the model loads, or the worker answers, before any vector goes
        for db in [catalog.store(c) for c in catalog.collections()]:  # every one opens, or none changes
            _forget_vectors(db)
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
    if not reset:  # prepared before the reset
        models.prepare()
    with console.Progress(total, "chunks") as progress:
        done, broken = _embed(models, [db for _, db, n in pending if n], using, progress)
    console.say(f"embedded {done - broken:,} chunks in {console.duration(progress.elapsed)}")
    if broken:
        console.say(f"{console.count(broken, 'chunk')} could not be embedded (the model returned NaN); try again")


def _embed(
    models: _models.Models, stores: list[sqlite3.Connection], using: str, progress: console.Progress
) -> tuple[int, int]:
    """Embed the slabs of these stores as the reader brings them, and hand each to the writer once all its vectors
    are in. Returns (chunks done, chunks the model broke)."""
    turn = threading.Lock()  # the store's connection, which the reader and the writer take in turn
    # (slab, part) of each batch handed to the model, not yet back
    on_gpu: collections.deque[tuple[_Slab, list[int]]] = collections.deque()
    done = 0
    tally = _Tally(stores)
    with _Reader(models, stores, turn, tally) as slabs, _Writer(turn, using, tally) as writer:

        def batches() -> Iterator[list[str]]:
            for slab in slabs:
                for part in slab.parts:
                    on_gpu.append((slab, part))
                    yield [slab.texts[i] for i in part]

        for vectors in models.encode_batches(batches()):
            slab, part = on_gpu.popleft()
            if (full := slab.add(part, vectors)) is not None:
                writer.write(slab, full)
            done += len(part)
            progress.update(done, tally.total)
    return done, writer.broken


class _Tally:
    """The chunks to embed in all, as embedding goes: those read so far, and those the stores' counts say are unread
    still, so that chunks indexed or deleted meanwhile (`sift index` beside `sift embed`) count. The reader counts anew
    after every slab it reads, the writer adds the vectors it stored, both under the stores' turn; the progress reads
    `total` as it is. Never less than the chunks read: the counts miss a chunk read and deleted before its vector
    is stored, until the reader moves on."""

    def __init__(self, stores: Iterable[sqlite3.Connection]) -> None:
        stores = list(stores)
        self.read, self.stored, self.reading = dict.fromkeys(stores, 0), dict.fromkeys(stores, 0), set(stores)
        self.total = 0
        self._count()

    def took(self, db: sqlite3.Connection, rows: list[Passage]) -> None:
        """The reader read these rows of a store, none when it has read all."""
        self.read[db] += len(rows)
        if not rows:
            self.reading.discard(db)
        self._count()

    def wrote(self, db: sqlite3.Connection, stored: int) -> None:
        self.stored[db] += stored

    def _count(self) -> None:
        """A store's counts hold every chunk without a vector, those read and not stored yet too."""
        unread = (max(_pending(db) - self.read[db] + self.stored[db], 0) for db in self.reading)
        self.total = sum(self.read.values()) + sum(unread)


class _Slab:
    """Chunks embedded and written together: their ids and texts, in batches of similar length (parts: indexes into
    texts), and their vectors as they come."""

    def __init__(self, db: sqlite3.Connection, ids: list[int], texts: list[str], parts: list[list[int]]) -> None:
        self.db, self.ids, self.texts, self.parts = db, ids, texts, parts
        self.vectors: Vectors | None = None
        self._left = len(texts)

    def add(self, part: list[int], vectors: Vectors) -> Vectors | None:
        """Take the vectors of a part; returns the slab's vectors once all are in."""
        if self.vectors is None:
            self.vectors = np.empty((len(self.texts), vectors.shape[1]), np.float16)
        self.vectors[part] = vectors
        self._left -= len(part)
        return None if self._left else self.vectors


class _End(Enum):
    DONE = 1


_DONE = _End.DONE


class _Reader:
    """A thread that reads the slabs still to embed, AHEAD at most, sorted by length; iterated, the slabs as they come.
    An error in the thread is raised where they're taken."""

    def __init__(
        self, models: _models.Models, stores: list[sqlite3.Connection], turn: threading.Lock, tally: _Tally
    ) -> None:
        self.models, self.stores, self.turn, self.tally = models, stores, turn, tally
        self.slabs: queue.Queue[_Slab | BaseException | _End] = queue.Queue(AHEAD)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read, name="sift-embed-read", daemon=True)

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        self.thread.join()

    def __iter__(self) -> Iterator[_Slab]:
        while (slab := self.slabs.get()) is not _DONE:
            if isinstance(slab, BaseException):
                raise slab
            yield slab

    def _read(self) -> None:
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

    def _rows(self, db: sqlite3.Connection, last: int) -> list[Passage]:
        """The next slab of a store's chunks without a vector, after the chunk `last`. The chunks and their texts are
        read as they are at one moment: a `sift index` deleting them in between would leave a slab of none, which
        ends the store's."""
        with self.turn, store.snapshot([db]):
            rows = passages(db, f"{PENDING} AND c.id > ? ORDER BY c.id LIMIT ?", (last, SLAB))
            self.tally.took(db, rows)
            return rows

    def _put(self, item: _Slab | BaseException | _End) -> None:
        """Queue an item, unless embedding stops meanwhile."""
        while not self.stop.is_set():
            try:
                self.slabs.put(item, timeout=0.1)
                return
            except queue.Full:
                continue


class _Writer:
    """A thread that writes the vectors of each slab handed to it, in order, and adds up the chunks the model broke.
    An error in it is raised at the next slab handed over, or at the end."""

    def __init__(self, turn: threading.Lock, model: str, tally: _Tally) -> None:
        self.turn, self.model, self.tally, self.broken = turn, model, tally, 0
        self._writes: list[Future[int]] = []
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="sift-embed-write")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, kind: type[BaseException] | None, *exc: object) -> None:
        self._pool.shutdown(wait=True)  # what was embedded is written, even when embedding stops
        if kind is None:
            self._settle(self._writes)

    def write(self, slab: _Slab, vectors: Vectors) -> None:
        self._settle([w for w in self._writes if w.done()])
        self._writes.append(self._pool.submit(self._store, slab, vectors))

    def _store(self, slab: _Slab, vectors: Vectors) -> int:
        with self.turn:
            try:
                stored, broken = _store_vectors(slab.db, slab.ids, vectors, self.model)
            except BaseException:
                slab.db.rollback()  # none of the slab, rather than part of it in the next slab's commit
                raise
            self.tally.wrote(slab.db, stored)
            return broken

    def _settle(self, writes: list[Future[int]]) -> None:
        for write in list(writes):
            self.broken += write.result()
            self._writes.remove(write)


def _chosen(catalog: store.Catalog[C], names: Sequence[str] | None) -> list[C]:
    """The collections named, or every active one. An unknown name is an error that names those there are."""
    if names:
        return [catalog.get(name) for name in dict.fromkeys(names)]
    return [c for c in catalog.collections() if c.active]


def _stores(catalog: store.Catalog[C], collections: Iterable[C], named: bool) -> Generator[tuple[C, store.Connection]]:
    """The collections with their stores: one whose store is missing is passed over, as `sift index` does, unless
    `named`, which makes it an error."""
    for c in collections:
        try:
            yield c, catalog.opened(c)
        except LookupError as missing:
            if named:
                raise
            console.say(f"{missing}; skipped")


def _forget_vectors(db: sqlite3.Connection) -> None:
    store.drop_vectors(db, "1")
    db.execute("DELETE FROM meta WHERE key IN ('model', 'dim', 'kw_ordered')")  # kw_ordered: see postings._unordered
    store.touch(db)
    db.commit()


def _check_model(db: sqlite3.Connection, name: str, using: str) -> None:
    """A collection embedded already must have been embedded with the model in use."""
    model = store.get_meta(db, "model")
    if model and model != using:
        raise SystemExit(f"{name} was embedded with {model}; run `sift embed --reset` to switch to {using}")


def _drop_broken_vectors(db: sqlite3.Connection) -> None:
    """sift 0.5 and older embedded in fp16, where EmbeddingGemma overflows now and then into NaN vectors. Drop those,
    once per store, so they are embedded again."""
    if store.get_meta(db, "vectors_checked"):
        return
    if n := store.counts(db)["vectors"]:
        console.say(f"checking {n:,} vectors for broken ones (once) ...")
        broken: list[int] = []
        cursor = db.execute("SELECT chunk, vec FROM vectors")
        while rows := cursor.fetchmany(100_000):
            vectors = np.frombuffer(b"".join(vec for _, vec in rows), np.float16).reshape(len(rows), -1)
            broken += [rows[i][0] for i in np.flatnonzero(~np.isfinite(vectors).all(axis=1))]
        if broken:
            store.drop_vectors(db, "chunk IN (SELECT value FROM json_each(?))", (json.dumps(broken),))
            store.touch(db)
            console.say(f"{console.count(len(broken), 'broken vector')} dropped, to be embedded again")
    store.set_meta(db, "vectors_checked", 1)
    db.commit()


def by_length(models: _models.Models, texts: list[str]) -> list[list[int]]:
    """The texts in batches of similar length to the model, longest first, as lists of indexes into them: each batch
    is padded to its longest text, and characters tell tokens poorly."""
    lengths, step = models.lengths(texts), models.batch
    order = sorted(range(len(texts)), key=lambda i: -lengths[i])
    return [order[start : start + step] for start in range(0, len(order), step)]


def encode(models: _models.Models, texts: list[str]) -> Vectors:
    """Vectors for texts, embedded in batches of similar length (by_length), one call to the models each; for no
    texts none, and no call."""
    vectors: Vectors | None = None
    for part in by_length(models, texts):
        out = models.encode([texts[i] for i in part])
        if vectors is None:
            vectors = np.empty((len(texts), out.shape[1]), np.float16)
        vectors[part] = out
    return np.empty((0, 0), np.float16) if vectors is None else vectors


def _store_vectors(db: sqlite3.Connection, chunks: list[int], vectors: Vectors, model: str) -> tuple[int, int]:
    """Store the intact vectors of the chunks still there, and commit. A NaN vector, an overflow inside the model,
    stays out, so its chunk stays pending; so does the vector of a chunk another sift deleted meanwhile, which would
    count as the vector of the chunk that took its place. Returns how many vectors it stored, and how many the model
    broke."""
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
    return int(keep.sum()), int((~intact).sum())
