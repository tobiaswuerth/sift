"""`sift embed`: embedding every chunk of the index that has no vector yet, slab by slab, with progress. The models
come from outside (`models.default()` unless given), so this runs the same on a GPU, a CPU or through a worker."""

import numpy as np

from . import console, store
from . import models as _models

SLAB = 20000  # chunks per commit; length sorting happens within a slab


def _text(seq, title, text):
    """What gets embedded: chunks after the first repeat their document title, unless they already start with it."""
    return text if seq == 0 or text.lstrip("# ").startswith(title) else f"{title}\n\n{text}"


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
        console.say("nothing to embed")
        return
    console.say(f"embedding {total:,} chunks with {using} on {models.address}")
    models.prepare(lambda: _longest_pending(db))
    done = broken = last = 0
    slab = f"SELECT c.id, c.seq, d.title, substr(d.text, c.pos + 1, c.len) FROM {PENDING} AND c.id > ? "
    with console.Progress(total, "chunks") as progress:
        while rows := db.execute(slab + "ORDER BY c.id LIMIT ?", (last, SLAB)).fetchall():
            texts = [_text(*row[1:]) for row in rows]
            vectors = _embed(models, texts, lambda n, before=done: progress.update(before + n))
            broken += _store_vectors(db, [row[0] for row in rows], vectors, using)
            done, last = done + len(rows), rows[-1][0]
    console.say(f"embedded {done - broken:,} chunks in {console.duration(progress.elapsed)}")
    if broken:
        console.say(f"{console.count(broken, 'chunk')} could not be embedded (the model returned NaN); try again")


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
        console.say(f"checking {n:,} vectors for broken ones (once) ...")
        broken, cursor = [], db.execute("SELECT chunk, vec FROM vectors")
        while rows := cursor.fetchmany(100_000):
            vectors = np.frombuffer(b"".join(vec for _, vec in rows), np.float16).reshape(len(rows), -1)
            broken += [rows[i][0] for i in np.flatnonzero(~np.isfinite(vectors).all(axis=1))]
        if broken:
            db.executemany("DELETE FROM vectors WHERE chunk = ?", [(chunk,) for chunk in broken])
            store.touch(db)
            console.say(f"{console.count(len(broken), 'broken vector')} dropped, to be embedded again")
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
