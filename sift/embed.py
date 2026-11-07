"""`sift embed`: embedding every chunk that has no vector yet, collection by collection and slab by slab, with progress.
The models come from outside (`models.default()` unless given), so this runs the same on a GPU, a CPU or through a
worker."""

import numpy as np

from . import console, store
from . import models as _models

SLAB = 20000  # chunks per commit; length sorting happens within a slab


def text_of(seq, title, text):
    """What gets embedded: chunks after the first repeat their document title, unless they already start with it."""
    return text if seq == 0 or text.lstrip("# ").startswith(title) else f"{title}\n\n{text}"


PENDING = "chunks c JOIN docs d ON d.id = c.doc WHERE NOT EXISTS (SELECT 1 FROM vectors v WHERE v.chunk = c.id)"
"""The chunks still without a vector, as a FROM clause."""


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
    pending = [(c, db, db.execute(f"SELECT count(*) FROM {PENDING}").fetchone()[0]) for c, db in stores]
    total, named = sum(n for *_, n in pending), ", ".join(collections or ())
    if not total:
        console.say(f"nothing to embed in {named}" if named else "nothing to embed")
        return
    of = f" of {named}" if named else ""
    console.say(f"embedding {console.count(total, 'chunk')}{of} with {using} on {models.address}")
    models.prepare()
    done = broken = 0
    slab = f"SELECT c.id, c.seq, d.title, substr(d.text, c.pos + 1, c.len) FROM {PENDING} AND c.id > ? "
    with console.Progress(total, "chunks") as progress:
        for _, db, n in pending:
            last = -1
            while n and (rows := db.execute(slab + "ORDER BY c.id LIMIT ?", (last, SLAB)).fetchall()):
                texts = [text_of(*row[1:]) for row in rows]
                vectors = encode(models, texts, lambda k, before=done: progress.update(before + k))
                broken += _store_vectors(db, [row[0] for row in rows], vectors, using)
                done, last = done + len(rows), rows[-1][0]
    console.say(f"embedded {done - broken:,} chunks in {console.duration(progress.elapsed)}")
    if broken:
        console.say(f"{console.count(broken, 'chunk')} could not be embedded (the model returned NaN); try again")


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
    db.execute("DELETE FROM vectors")
    db.execute("DELETE FROM meta WHERE key IN ('model', 'dim')")
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


def encode(models, texts, report=None):
    """Vectors for texts, longest first in batches of similar length to the model: each batch is padded to its longest
    text, and characters tell tokens poorly. report(n), if given, after each batch, n texts done."""
    lengths, step, vectors = models.lengths(texts), models.batch, None
    order = sorted(range(len(texts)), key=lambda i: -lengths[i])
    for start in range(0, len(order), step):
        part = order[start : start + step]
        out = models.encode([texts[i] for i in part])
        if vectors is None:
            vectors = np.empty((len(texts), out.shape[1]), np.float16)
        vectors[part] = out
        if report:
            report(start + len(part))
    return vectors


def _store_vectors(db, chunks, vectors, model):
    """Store the intact vectors and commit. A NaN vector, an overflow inside the model, stays out, so its chunk
    stays pending. Returns how many stayed out."""
    intact = np.isfinite(vectors).all(axis=1)
    store.put_vectors(db, [chunk for chunk, ok in zip(chunks, intact) if ok], vectors[intact])
    store.set_meta(db, "model", model)
    store.set_meta(db, "dim", vectors.shape[1])
    store.touch(db)  # tells a running server to reload the codes
    db.commit()
    return int((~intact).sum())
