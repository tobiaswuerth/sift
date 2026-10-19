"""Embedding and reranking models, and the batched GPU embedding pipeline."""

import contextlib
import logging
import os
import sys
from functools import cache
from time import monotonic, perf_counter

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
IDLE_SECONDS = 300  # unused this long, models and vectors give their GPU memory back

_batch_size = 128  # until calibrated on CUDA; a good size for CPUs and Apple GPUs
_resident = {}  # what is on the GPU now: key -> function that moves it back to RAM
_released = False  # whether everything left the GPU at some point, so the budget needs a fresh look
_last_use = 0.0


def gpu():
    """Whether sift computes on a GPU (NVIDIA or Apple)."""
    return torch.device(DEVICE).type in ("cuda", "mps")


def fp16():
    """Whether models and vectors use fp16: yes on GPUs, no on CPUs, which are slow at fp16 matmul."""
    return gpu()


def _free_gpu_memory():
    """Free and total memory of the GPU. On Windows (WDDM), CUDA counts memory other programs hold as free,
    since the driver could page it out; NVML, which nvidia-smi reads, does not. The smaller value wins."""
    free, total = torch.cuda.mem_get_info(DEVICE)
    with contextlib.suppress(Exception):  # NVML missing or unusable: CUDA's figure is all there is
        free = min(free, total - torch.cuda.device_memory_used(DEVICE))
    return free, total


def cuda_pytorch_command(platform=sys.platform):
    """How to replace a CPU-only PyTorch with the CUDA build. Windows wheels bundle their CUDA libraries, so
    torch's other dependencies can stay; Linux wheels depend on nvidia-* packages, which must come along."""
    no_deps = "--no-deps " if platform == "win32" else ""
    return f"pip install --force-reinstall {no_deps}torch --index-url https://download.pytorch.org/whl/cu128"


def _unused_nvidia_gpu():
    """Why an NVIDIA GPU sits unused, if there is one; None otherwise."""
    try:
        import pynvml

        pynvml.nvmlInit()
        if not pynvml.nvmlDeviceGetCount():
            return None
    except Exception:  # no NVML, no driver, no GPU
        return None
    if torch.version.cuda is None:
        return (
            "found an NVIDIA GPU, but this PyTorch build has no CUDA support, so sift runs on the CPU (much slower). "
            "To use the GPU, reinstall with uv: `uv tool install --reinstall --torch-backend auto sift-rag` "
            f"(or run `uvx --torch-backend auto sift-rag ...`), or with pip: `{cuda_pytorch_command()}`"
        )
    return f"found an NVIDIA GPU, but PyTorch (CUDA {torch.version.cuda}) can't use it; update the NVIDIA driver"


@cache
def limit_gpu_memory():
    """Cap PyTorch's GPU memory at what is free now, minus headroom for other programs, and return that budget.

    Past the cap, PyTorch raises out-of-memory inside sift instead of taking memory that other programs need.
    With too little free memory, sift runs on the CPU instead.
    """
    global DEVICE
    if torch.device(DEVICE).type != "cuda":
        if DEVICE == "cpu" and (reason := _unused_nvidia_gpu()):
            print(reason, file=sys.stderr)
        return None
    budget = _set_gpu_budget()
    if budget is None:
        DEVICE = "cpu"
    return budget


def _set_gpu_budget():
    """Cap PyTorch's GPU memory at what other programs leave free, minus headroom. None if too little is free."""
    free, total = _free_gpu_memory()
    budget = free - max(HEADROOM * total, GiB)
    if budget < MIN_BUDGET:
        print(f"only {free / GiB:.1f} GB of GPU memory is free; using the CPU (slower)", file=sys.stderr)
        return None
    index = torch.device(DEVICE).index  # PyTorch wants an index here; plain "cuda" means the current device
    torch.cuda.set_per_process_memory_fraction(budget / total, torch.cuda.current_device() if index is None else index)
    print(f"using at most {budget / GiB:.1f} of {total / GiB:.1f} GB of GPU memory", file=sys.stderr)
    return budget


def acquire(key, to_gpu, off_gpu):
    """Make sure something is on the GPU: to_gpu() moves it there unless it is already, off_gpu() moves it back.

    Models and vectors wait in RAM and come to the GPU when used; every call counts as use. On the CPU, nothing moves.
    """
    global _last_use
    _last_use = monotonic()
    _return_to_gpu()
    if not gpu() or key in _resident:
        return
    to_gpu()
    _resident[key] = off_gpu


def _return_to_gpu():
    """After an idle release, look at the GPU memory again: other programs may hold more or less of it by now.
    With too little left, sift uses the CPU from then on, as it does when there is too little at start-up."""
    global DEVICE, _released
    if not (_released and torch.device(DEVICE).type == "cuda"):
        return
    _released = False
    if _set_gpu_budget() is None:
        DEVICE = "cpu"
        embedder.cache_clear()  # the models load again, in float32 for the CPU
        reranker.cache_clear()


def discard(key):
    """Forget something that left the GPU on its own, such as vectors replaced by a reload."""
    _resident.pop(key, None)


def release_idle():
    """Move models and vectors back to RAM once the GPU has been unused for IDLE_SECONDS. Returns whether it did."""
    global _released
    if not _resident or monotonic() < _last_use + IDLE_SECONDS:  # not now - _last_use, which can round below it
        return False
    for off_gpu in _resident.values():
        off_gpu()
    _resident.clear()
    _released = True
    backend = {"cuda": torch.cuda, "mps": torch.mps}.get(torch.device(DEVICE).type)
    if backend:
        backend.empty_cache()  # hand the memory back to the GPU, not just to PyTorch's cache
    print(f"idle for {IDLE_SECONDS // 60} min: models and vectors moved off the GPU", file=sys.stderr)
    return True


def _on_gpu(load):
    """The model load() returns, on the GPU while in use. The check comes first: it may switch sift to the CPU,
    and then load() returns a model for the CPU."""
    _return_to_gpu()
    model = load()
    acquire(id(model), lambda: model.to(DEVICE), lambda: model.to("cpu"))
    return model


@cache
def embedder():
    from sentence_transformers import SentenceTransformer

    limit_gpu_memory()
    model = SentenceTransformer(EMBED_MODEL, device="cpu")  # into RAM; it moves to the GPU when used
    if fp16():
        model.half()
    model.max_seq_length = min(model.max_seq_length or MAX_TOKENS, MAX_TOKENS)
    return model


@cache
def reranker():
    from sentence_transformers import CrossEncoder

    limit_gpu_memory()
    model = CrossEncoder(RERANK_MODEL, device="cpu", max_length=MAX_TOKENS)  # into RAM, like the embedder
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
    model = _on_gpu(embedder)
    budget = limit_gpu_memory()
    if budget is None or torch.device(DEVICE).type != "cuda":  # no GPU, or given up
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
    per_text = max((used - one) / (n - 1) if n > 1 else one, 1)  # tiny texts may not change the peak at all
    _batch_size = max(1, min(failed - 1, MAX_BATCH, 1 + int((target - one) / per_text)))
    print(f"batch size {_batch_size}, calibrated to the GPU memory budget", file=sys.stderr)
    return _batch_size


def encode(texts, query=False):
    """Normalised fp16 embeddings, one row per text. Sentence Transformers sorts by length to minimise padding."""
    model = _on_gpu(embedder)
    fn = model.encode_query if query else model.encode_document
    return _oom_safe(
        lambda t, bs: fn(t, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
        texts,
    ).astype(np.float16)


def rerank(query, texts):
    model = _on_gpu(reranker)
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
        """SELECT count(*) FROM chunks c JOIN fts ON fts.rowid = c.id
           WHERE NOT EXISTS (SELECT 1 FROM vectors v WHERE v.chunk = c.id)"""
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
    calibrate_batch_size(max((_text(*row) for row in candidates), key=len, default=None))  # None: embedded meanwhile
    done, last, t0 = 0, 0, perf_counter()

    def progress(n):
        rate = n / max(perf_counter() - t0, 1e-6)  # coarse clocks can return the same reading twice
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
    print(f"\rembedded {done} chunks in {(perf_counter() - t0) / 60:.1f} min" + " " * 20, file=sys.stderr)
