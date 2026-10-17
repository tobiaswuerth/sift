"""Embedding and reranking models, and the batched GPU embedding pipeline."""

import os
import sys
import time
from functools import cache

import numpy as np
import torch

from . import store


def _device():
    """SIFT_DEVICE if set, else an NVIDIA GPU (CUDA), else an Apple GPU (MPS), else the CPU."""
    if os.environ.get("SIFT_DEVICE"):
        return os.environ["SIFT_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


EMBED_MODEL = os.environ.get("SIFT_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
RERANK_MODEL = os.environ.get("SIFT_RERANK_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base")
DEVICE = _device()
MAX_TOKENS = 1024
SLAB = 20000  # chunks per commit; length sorting happens within a slab

_batch_size = int(os.environ.get("SIFT_BATCH_SIZE", "128"))


def fp16():
    """Whether models and vectors use fp16: yes on GPUs, no on CPUs, which are slow at fp16 matmul."""
    return torch.device(DEVICE).type in ("cuda", "mps")


@cache
def embedder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    if fp16():
        model.half()
    model.max_seq_length = min(model.max_seq_length or MAX_TOKENS, MAX_TOKENS)
    return model


@cache
def reranker():
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(RERANK_MODEL, device=DEVICE, max_length=MAX_TOKENS)
    if fp16():
        model.model.half()
    return model


def _oom_safe(fn, items):
    """Run fn(items, batch_size), halving the batch size when the GPU runs out of memory."""
    global _batch_size
    while True:
        try:
            return fn(items, _batch_size)
        except RuntimeError as e:  # CUDA raises OutOfMemoryError, a RuntimeError; MPS a plain RuntimeError
            cuda = isinstance(e, torch.cuda.OutOfMemoryError)
            if _batch_size == 1 or not (cuda or "MPS backend out of memory" in str(e)):
                raise
            (torch.cuda if cuda else torch.mps).empty_cache()
            _batch_size //= 2
            print(f"\nout of GPU memory, batch size now {_batch_size}", file=sys.stderr)


def encode(texts, query=False):
    """Normalised fp16 embeddings, one row per text. Sentence Transformers sorts by length to minimise padding."""
    model = embedder()
    fn = model.encode_query if query else model.encode_document
    return _oom_safe(
        lambda t, bs: fn(t, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True), texts
    ).astype(np.float16)


def rerank(query, texts):
    model = reranker()
    return _oom_safe(lambda t, bs: model.predict([(query, x) for x in t], batch_size=bs), texts)


def embed_pending(db, reset=False):
    """Embed every chunk without a vector, slab by slab. Safe to interrupt and resume."""
    if reset:
        db.execute("DELETE FROM vectors")
        db.execute("DELETE FROM meta WHERE key IN ('model', 'dim')")
        store.touch(db)
        db.commit()
    model = store.get_meta(db, "model")
    if model and model != EMBED_MODEL:
        raise SystemExit(f"index was embedded with {model}; run `sift embed --reset` to switch to {EMBED_MODEL}")
    pending = """SELECT c.id, c.seq, fts.title, fts.text FROM chunks c JOIN fts ON fts.rowid = c.id
                 WHERE c.id > ? AND NOT EXISTS (SELECT 1 FROM vectors v WHERE v.chunk = c.id) ORDER BY c.id LIMIT ?"""
    total = db.execute(
        "SELECT count(*) FROM chunks c WHERE NOT EXISTS (SELECT 1 FROM vectors v WHERE v.chunk = c.id)"
    ).fetchone()[0]
    if not total:
        print("nothing to embed", file=sys.stderr)
        return
    embedder()
    done, last, t0 = 0, 0, time.perf_counter()
    while rows := db.execute(pending, (last, SLAB)).fetchall():
        # chunks after the first repeat their document title, unless they already start with it
        texts = [
            text if seq == 0 or text.lstrip("# ").startswith(title) else f"{title}\n\n{text}"
            for _, seq, title, text in rows
        ]
        vecs = encode(texts)
        db.executemany(
            "INSERT OR REPLACE INTO vectors(chunk, vec) VALUES(?, ?)", [(r[0], v.tobytes()) for r, v in zip(rows, vecs)]
        )
        store.set_meta(db, "model", EMBED_MODEL)
        store.set_meta(db, "dim", vecs.shape[1])
        store.touch(db)  # tells a running server to reload vectors
        db.commit()
        done, last = done + len(rows), rows[-1][0]
        rate = done / (time.perf_counter() - t0)
        print(
            f"\rembedded {done}/{total} chunks, {rate:.0f}/s, {max(total - done, 0) / rate / 60:.1f} min left ",
            end="",
            file=sys.stderr,
            flush=True,
        )
    print(f"\rembedded {done} chunks in {(time.perf_counter() - t0) / 60:.1f} min" + " " * 20, file=sys.stderr)
