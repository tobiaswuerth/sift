"""Embedding and reranking models, and the batched GPU embedding pipeline."""

import contextlib
import logging
import os
import sys
import threading
from functools import cache
from time import monotonic, perf_counter

import numpy as np
import torch

from . import remote, store

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


EMBED_MODEL = os.environ.get("SIFT_EMBED_MODEL", "google/embeddinggemma-2")
RERANK_MODEL = os.environ.get("SIFT_RERANK_MODEL", "cross-encoder/ettin-reranker-150m-v1")
DEVICE = _device()
MAX_TOKENS = 1024
SLAB = 20000  # chunks per commit; length sorting happens within a slab
HEADROOM = 0.1  # share of the GPU left free for other programs, at least 1 GiB
MIN_BUDGET = 2 * GiB  # with less GPU memory than this, the models run on the CPU
FILL = 0.9  # share of the memory budget a calibrated batch may use
MAX_BATCH = 1024
PROGRESS_SECONDS = 1  # the progress line redraws itself this often, so a slow batch still shows sift at work
IDLE_SECONDS = 300  # unused this long, models and vectors give their GPU memory back

_batch_size = 128  # until calibrated on CUDA; a good size for CPUs and Apple GPUs
_resident = {}  # what is on the GPU now: key -> function that moves it back to RAM
_released = False  # whether everything left the GPU at some point, so the budget needs a fresh look
_last_use = 0.0


def gpu():
    """Whether sift computes on a GPU (NVIDIA or Apple)."""
    return torch.device(DEVICE).type in ("cuda", "mps")


def fp16():
    """Whether vectors are searched in fp16: yes on GPUs, no on CPUs, which are slow at fp16 matmul."""
    return gpu()


def half_precision():
    """The 16-bit type the models compute in: bfloat16 on GPUs that have it, None (float32) elsewhere. Never fp16:
    its range is too small for EmbeddingGemma, which overflows in it now and then into NaN vectors, nor for Qwen
    models; bfloat16 has the range of float32."""
    return torch.bfloat16 if torch.device(DEVICE).type == "cuda" and torch.cuda.is_bf16_supported() else None


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
    return (
        f"pip install --force-reinstall {no_deps}torch torchvision --index-url https://download.pytorch.org/whl/cu128"
    )


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
    if dtype := half_precision():
        model.to(dtype)
    model.max_seq_length = min(model.max_seq_length or MAX_TOKENS, MAX_TOKENS)
    return model


@cache
def reranker():
    from sentence_transformers import CrossEncoder

    limit_gpu_memory()
    model = CrossEncoder(RERANK_MODEL, device="cpu", max_length=MAX_TOKENS)  # into RAM, like the embedder
    if dtype := half_precision():
        model.to(dtype)  # the whole model: its scoring head too, or PyTorch refuses to mix the two
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


def model_name():
    """The embedding model in use: the worker's, if there is one."""
    return remote.model()[0] if remote.REMOTE else EMBED_MODEL


def encode(texts, query=False):
    """Normalised fp16 embeddings, one row per text: from the worker, if there is one, else made here."""
    return remote.encode(texts, query) if remote.REMOTE else encode_here(texts, query)


def encode_here(texts, query=False):
    """Embeddings made on this machine. Sentence Transformers sorts by length to minimise padding."""
    model = _on_gpu(embedder)
    fn = model.encode_query if query else model.encode_document
    return _oom_safe(
        lambda t, bs: fn(t, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
        texts,
    ).astype(np.float16)


def rerank(query, texts):
    """Scores of (query, text) pairs, higher for better matches: from the worker, if there is one, else made here."""
    return remote.rerank(query, texts) if remote.REMOTE else rerank_here(query, texts)


def rerank_here(query, texts):
    """Reranking scores computed on this machine."""
    model = _on_gpu(reranker)
    return _oom_safe(
        lambda t, bs: model.predict([(query, x) for x in t], batch_size=bs, show_progress_bar=False), texts
    )


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


def embed_pending(db, reset=False):
    """Embed every chunk without a vector, slab by slab. Safe to interrupt and resume."""
    if reset:
        _forget_vectors(db)
    using = _model_for(db)
    _drop_broken_vectors(db)
    total = db.execute(f"SELECT count(*) FROM {PENDING}").fetchone()[0]
    if not total:
        print("nothing to embed", file=sys.stderr)
        return
    where = remote.address() if remote.REMOTE else DEVICE
    print(f"embedding {total:,} chunks with {using} on {where}", file=sys.stderr)
    if not remote.REMOTE:  # a worker sizes its batches itself
        calibrate_batch_size(_longest_pending(db))
    done = broken = last = 0
    slab = f"SELECT c.id, c.seq, d.title, substr(d.text, c.pos + 1, c.len) FROM {PENDING} AND c.id > ? "
    with _Progress(total) as progress:
        while rows := db.execute(slab + "ORDER BY c.id LIMIT ?", (last, SLAB)).fetchall():
            vectors = _embed([_text(*row[1:]) for row in rows], lambda n, before=done: progress.update(before + n))
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


def _model_for(db):
    """The embedding model in use, which must be the one the index was embedded with."""
    model, using = store.get_meta(db, "model"), model_name()
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


def _embed(texts, report):
    """Vectors for texts, longest first in batches of similar length; report(n) after each batch, n texts done."""
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    step, vectors = remote.BATCH if remote.REMOTE else _batch_size, None
    for start in range(0, len(order), step):
        part = order[start : start + step]
        out = encode([texts[i] for i in part])
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
