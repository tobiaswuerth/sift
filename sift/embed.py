"""Embedding and reranking models, and the batched GPU embedding pipeline."""

import logging
import os
import sys
import time
from functools import cache

import numpy as np
import torch

from . import store

# Loading Sentence Transformers makes PyTorch log a harmless note about distributed training on Windows and macOS.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
# The MCP server logs at INFO, where loading a model logs every HTTP request to Hugging Face.
for _name in ("httpx", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(_name).setLevel(logging.WARNING)

GiB = 2**30


def _device():
    """An NVIDIA GPU (CUDA) if there is one, else an Apple GPU (MPS), else the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


EMBED_MODEL = os.environ.get("SIFT_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
RERANK_MODEL = os.environ.get("SIFT_RERANK_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base")
DEVICE = _device()
MAX_TOKENS = 1024
SLAB = 20000  # chunks per commit; length sorting happens within a slab
HEADROOM = 0.1  # share of the GPU left free for other programs, at least 1 GiB
MIN_BUDGET = 2 * GiB  # with less GPU memory than this, the models run on the CPU
FILL = 0.9  # share of the memory budget a calibrated batch may use
MAX_BATCH = 1024
PROGRESS_BATCHES = 8  # report progress after this many batches

_batch_size = 128  # until calibrated on CUDA; a good size for CPUs and Apple GPUs


def fp16():
    """Whether models and vectors use fp16: yes on GPUs, no on CPUs, which are slow at fp16 matmul."""
    return torch.device(DEVICE).type in ("cuda", "mps")


def _free_gpu_memory():
    """Free and total memory of the GPU. On Windows (WDDM), CUDA counts memory other programs hold as free,
    since the driver could page it out; NVML, which nvidia-smi reads, does not. The smaller value wins."""
    free, total = torch.cuda.mem_get_info(DEVICE)
    try:
        free = min(free, total - torch.cuda.device_memory_used(DEVICE))
    except Exception:  # NVML missing or unusable: CUDA's figure is all there is
        pass
    return free, total


@cache
def limit_gpu_memory():
    """Cap PyTorch's GPU memory at what is free now, minus headroom for other programs, and return that budget.

    Past the cap, PyTorch raises out-of-memory inside sift instead of taking memory that other programs need.
    With too little free memory, sift runs on the CPU instead.
    """
    global DEVICE
    if torch.device(DEVICE).type != "cuda":
        return None
    free, total = _free_gpu_memory()
    budget = free - max(HEADROOM * total, GiB)
    if budget < MIN_BUDGET:
        print(f"only {free / GiB:.1f} GB of GPU memory is free; using the CPU (slower)", file=sys.stderr)
        DEVICE = "cpu"
        return None
    index = torch.device(DEVICE).index  # PyTorch wants an index here; plain "cuda" means the current device
    torch.cuda.set_per_process_memory_fraction(budget / total, torch.cuda.current_device() if index is None else index)
    print(f"using at most {budget / GiB:.1f} of {total / GiB:.1f} GB of GPU memory", file=sys.stderr)
    return budget


@cache
def embedder():
    from sentence_transformers import SentenceTransformer

    limit_gpu_memory()
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    if fp16():
        model.half()
    model.max_seq_length = min(model.max_seq_length or MAX_TOKENS, MAX_TOKENS)
    return model


@cache
def reranker():
    from sentence_transformers import CrossEncoder

    limit_gpu_memory()
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


def calibrate_batch_size(sample=None):
    """Measure the largest batch that fits the GPU memory budget, and use it from now on.

    Batches of the sample, which should be the longest text to embed (default: a text of the model's maximum
    length), run with sizes 1, 2, 4, ... until one uses half of the budget. Memory grows linearly with the batch
    size, so the last measurement predicts the size that fills FILL of the budget. A size that ran out of memory
    is never chosen.
    """
    global _batch_size
    model = embedder()
    budget = limit_gpu_memory()
    if budget is None:
        return _batch_size
    longest = sample or "x " * (4 * model.max_seq_length)  # the default is truncated to the longest input
    base = torch.cuda.memory_allocated(DEVICE)
    target = FILL * budget - base

    def cost(n):
        torch.cuda.reset_peak_memory_stats(DEVICE)
        model.encode_document([longest] * n, batch_size=n, show_progress_bar=False)
        return torch.cuda.max_memory_allocated(DEVICE) - base

    one = used = cost(1)
    n, failed = 1, MAX_BATCH + 1
    while n < MAX_BATCH and used < target / 2:
        try:
            used = cost(2 * n)
        except torch.cuda.OutOfMemoryError:
            failed = 2 * n
            break
        n *= 2
    torch.cuda.empty_cache()
    per_text = (used - one) / (n - 1) if n > 1 else one
    _batch_size = max(1, min(failed - 1, MAX_BATCH, 1 + int((target - one) / per_text)))
    print(f"batch size {_batch_size}, calibrated to the GPU memory budget", file=sys.stderr)
    return _batch_size


def encode(texts, query=False):
    """Normalised fp16 embeddings, one row per text. Sentence Transformers sorts by length to minimise padding."""
    model = embedder()
    fn = model.encode_query if query else model.encode_document
    return _oom_safe(
        lambda t, bs: fn(t, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
        texts,
    ).astype(np.float16)


def rerank(query, texts):
    model = reranker()
    return _oom_safe(
        lambda t, bs: model.predict([(query, x) for x in t], batch_size=bs, show_progress_bar=False), texts
    )


def _text(seq, title, text):
    """What gets embedded: chunks after the first repeat their document title, unless they already start with it."""
    return text if seq == 0 or text.lstrip("# ").startswith(title) else f"{title}\n\n{text}"


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
    # calibrate on the longest text of this run: no batch can need more memory per text
    candidates = db.execute(
        """SELECT c.seq, fts.title, fts.text FROM chunks c JOIN fts ON fts.rowid = c.id
           WHERE NOT EXISTS (SELECT 1 FROM vectors v WHERE v.chunk = c.id)
           ORDER BY length(fts.text) + length(fts.title) DESC LIMIT 100"""
    ).fetchall()
    calibrate_batch_size(max((_text(*row) for row in candidates), key=len))
    done, last, t0 = 0, 0, time.perf_counter()

    def progress(n):
        rate = n / (time.perf_counter() - t0)
        left = max(total - n, 0) / rate / 60
        print(f"\rembedded {n}/{total} chunks, {rate:.0f}/s, {left:.1f} min left ", end="", file=sys.stderr, flush=True)

    while rows := db.execute(pending, (last, SLAB)).fetchall():
        texts = [_text(seq, title, text) for _, seq, title, text in rows]
        # longest first, in steps of a few batches: similar lengths per batch, and progress every few seconds
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        vecs, step = None, _batch_size * PROGRESS_BATCHES
        for start in range(0, len(order), step):
            part = order[start : start + step]
            out = encode([texts[i] for i in part])
            if vecs is None:
                vecs = np.empty((len(texts), out.shape[1]), np.float16)
            vecs[part] = out
            progress(done + start + len(part))
        db.executemany(
            "INSERT OR REPLACE INTO vectors(chunk, vec) VALUES(?, ?)", [(r[0], v.tobytes()) for r, v in zip(rows, vecs)]
        )
        store.set_meta(db, "model", EMBED_MODEL)
        store.set_meta(db, "dim", vecs.shape[1])
        store.touch(db)  # tells a running server to reload vectors
        db.commit()
        done, last = done + len(rows), rows[-1][0]
    print(f"\rembedded {done} chunks in {(time.perf_counter() - t0) / 60:.1f} min" + " " * 20, file=sys.stderr)
