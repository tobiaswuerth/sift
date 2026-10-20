"""`sift embed`: embedding every chunk of the index that has no vector yet, slab by slab, with progress. The models
come from outside (`models.default()` unless given), so this runs the same on a GPU, a CPU or through a worker."""

import sys
import threading
from time import perf_counter

import numpy as np

from . import models as _models
from . import store

SLAB = 20000  # chunks per commit; length sorting happens within a slab
PROGRESS_SECONDS = 1  # the progress line redraws itself this often, so a slow batch still shows sift at work


def _text(seq, title, text):
    """What gets embedded: chunks after the first repeat their document title, unless they already start with it."""
    return text if seq == 0 or text.lstrip("# ").startswith(title) else f"{title}\n\n{text}"


def _duration(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02}:{seconds % 60:02}"


class _Progress:
    """The progress line of `sift embed`: a bar, the chunks done, the speed, and the time taken and left. It redraws
    itself every PROGRESS_SECONDS, so the time keeps moving while a slow batch runs (minutes, on a CPU)."""

    def __init__(self, total):
        self.total, self.done, self.start, self._width = total, 0, perf_counter(), 0
        self._stop, self._lock = threading.Event(), threading.Lock()
        self._thread = threading.Thread(target=self._redraw, name="sift-progress", daemon=True)

    def __enter__(self):
        self.draw()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        print(file=sys.stderr)  # ends the line

    def _redraw(self):
        while not self._stop.wait(PROGRESS_SECONDS):
            self.draw()

    def update(self, done):
        self.done = done
        self.draw()

    def draw(self):
        seconds, share = perf_counter() - self.start, self.done / self.total
        line = f"[{'#' * round(20 * share):<20}] {share:4.0%} {self.done:,}/{self.total:,} chunks, "
        rate = self.done / seconds if seconds > 0 else 0  # coarse clocks can return the same reading twice
        if rate:
            line += f"{rate:.1f}/s, " if rate < 10 else f"{rate:,.0f}/s, "
        line += f"{_duration(seconds)} elapsed"
        if rate:
            line += f", {_duration((self.total - self.done) / rate)} left"
        with self._lock:
            self._width = max(self._width, len(line))  # blanks out what a longer line left behind
            print(f"\r{line:<{self._width}}", end="", file=sys.stderr, flush=True)


PENDING = "chunks c JOIN docs d ON d.id = c.doc WHERE NOT EXISTS (SELECT 1 FROM vectors v WHERE v.chunk = c.id)"
"""The chunks still without a vector, as a FROM clause."""


def embed_pending(db, reset=False, models=None):
    """Embed every chunk without a vector, slab by slab, with `models` (default: `models.default()`). Safe to
    interrupt and resume."""
    models = models or _models.default()
    if reset:
        _forget_vectors(db)
    using = _model_for(db, models)
    _drop_broken_vectors(db)
    total = db.execute(f"SELECT count(*) FROM {PENDING}").fetchone()[0]
    if not total:
        print("nothing to embed", file=sys.stderr)
        return
    print(f"embedding {total:,} chunks with {using} on {models.address}", file=sys.stderr)
    models.prepare(lambda: _longest_pending(db))
    done = broken = last = 0
    slab = f"SELECT c.id, c.seq, d.title, substr(d.text, c.pos + 1, c.len) FROM {PENDING} AND c.id > ? "
    with _Progress(total) as progress:
        while rows := db.execute(slab + "ORDER BY c.id LIMIT ?", (last, SLAB)).fetchall():
            texts = [_text(*row[1:]) for row in rows]
            vectors = _embed(models, texts, lambda n, before=done: progress.update(before + n))
            broken += _store_vectors(db, [row[0] for row in rows], vectors, using)
            done, last = done + len(rows), rows[-1][0]
    print(f"embedded {done - broken:,} chunks in {_duration(perf_counter() - progress.start)}", file=sys.stderr)
    if broken:
        print(f"{_count(broken, 'chunk')} could not be embedded (the model returned NaN); try again", file=sys.stderr)


def _count(n, noun):
    """'1 chunk', '2 chunks'."""
    return f"{n:,} {noun}{'' if n == 1 else 's'}"


def _forget_vectors(db):
    db.execute("DELETE FROM vectors")
    db.execute("DELETE FROM meta WHERE key IN ('model', 'dim')")
    store.touch(db)
    db.commit()


def _model_for(db, models):
    """The embedding model in use, which must be the one the index was embedded with."""
    model, using = store.get_meta(db, "model"), models.name
    if model and model != using:
        raise SystemExit(f"index was embedded with {model}; run `sift embed --reset` to switch to {using}")
    return using


def _drop_broken_vectors(db):
    """sift 0.5 and older embedded in fp16, where EmbeddingGemma overflows now and then into NaN vectors. Drop those,
    once per index, so they are embedded again."""
    if store.get_meta(db, "vectors_checked"):
        return
    if n := db.execute("SELECT count(*) FROM vectors").fetchone()[0]:
        print(f"checking {n:,} vectors for broken ones (once) ...", file=sys.stderr, flush=True)
        broken, cursor = [], db.execute("SELECT chunk, vec FROM vectors")
        while rows := cursor.fetchmany(100_000):
            vectors = np.frombuffer(b"".join(vec for _, vec in rows), np.float16).reshape(len(rows), -1)
            broken += [rows[i][0] for i in np.flatnonzero(~np.isfinite(vectors).all(axis=1))]
        if broken:
            db.executemany("DELETE FROM vectors WHERE chunk = ?", [(chunk,) for chunk in broken])
            store.touch(db)
            print(f"{_count(len(broken), 'broken vector')} dropped, to be embedded again", file=sys.stderr)
    store.set_meta(db, "vectors_checked", 1)
    db.commit()


def _longest_pending(db):
    """The longest text still to embed, to calibrate on: no batch can need more memory per text. None if none."""
    sql = (
        f"SELECT c.seq, d.title, substr(d.text, c.pos + 1, c.len) FROM {PENDING} ORDER BY c.len + length(d.title) DESC"
    )
    return max((_text(*row) for row in db.execute(sql + " LIMIT 100")), key=len, default=None)


def _embed(models, texts, report):
    """Vectors for texts, longest first in batches of similar length; report(n) after each batch, n texts done."""
    order, step, vectors = sorted(range(len(texts)), key=lambda i: -len(texts[i])), models.batch, None
    for start in range(0, len(order), step):
        part = order[start : start + step]
        out = models.encode([texts[i] for i in part])
        if vectors is None:
            vectors = np.empty((len(texts), out.shape[1]), np.float16)
        vectors[part] = out
        report(start + len(part))
    return vectors


def _store_vectors(db, chunks, vectors, model):
    """Store the intact vectors and commit. A NaN vector, an overflow inside the model, stays out, so its chunk
    stays pending. Returns how many stayed out."""
    intact = np.isfinite(vectors).all(axis=1)
    db.executemany(
        "INSERT OR REPLACE INTO vectors(chunk, vec) VALUES(?, ?)",
        [(chunk, vec.tobytes()) for chunk, vec, ok in zip(chunks, vectors, intact) if ok],
    )
    store.set_meta(db, "model", model)
    store.set_meta(db, "dim", vectors.shape[1])
    store.touch(db)  # tells a running server to reload vectors
    db.commit()
    return int((~intact).sum())
