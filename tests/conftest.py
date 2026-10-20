import functools
import re
import sys
import types
import zlib

import numpy as np
import pytest

from sift import indexing, store
from sift.accelerator import Accelerator

DIM = 64


def words(text):
    return re.findall(r"\w+", text.lower())


@functools.cache
def fake_vector(text):
    """Deterministic bag-of-words embedding: texts sharing words point in similar directions."""
    v = np.zeros(DIM, np.float32)
    for w in words(text):
        v[zlib.crc32(w.encode()) % DIM] += 1
    return v / (np.linalg.norm(v) or 1)


class FakeSentenceTransformer:
    max_batch = 10**6  # batches above this run out of GPU memory
    gpu = "cuda"  # whose out-of-memory error to raise: "cuda" or "mps"
    gpu_memory = None  # a FakeGpuMemory, consulted when the model runs on CUDA

    def __init__(self, name, device=None):
        self.name, self.device, self.max_seq_length, self.dtype = name, device, None, "float32"

    def half(self):
        self.dtype = "float16"
        return self

    def to(self, target):
        """A device or a dtype, as nn.Module.to takes them."""
        import torch

        if isinstance(target, torch.dtype):
            self.dtype = str(target).removeprefix("torch.")
        else:
            self.device = str(target)
        return self

    def encode_document(
        self, texts, batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=None
    ):
        import torch

        assert show_progress_bar is False, "sift must not draw progress bars (the server's log is no terminal)"
        if str(self.device).startswith("cuda") and self.gpu_memory:
            for i in range(0, len(texts), batch_size):
                self.gpu_memory.run(len(texts[i : i + batch_size]))
        if batch_size > self.max_batch and self.gpu == "mps":  # MPS raises a plain RuntimeError
            raise RuntimeError("MPS backend out of memory (MPS allocated: 17.00 GB, other allocations: 0 bytes)")
        if batch_size > self.max_batch:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return np.stack([fake_vector(t) for t in texts])

    encode_query = encode_document


class FakeCrossEncoder:
    """A transformer (.model) with a scoring head on top, like a real cross-encoder: converting only the transformer
    to another dtype leaves the head behind, and PyTorch refuses to mix the two."""

    def __init__(self, name, device=None, max_length=None):
        self.name, self.device, self.model, self.head = name, device, FakeSentenceTransformer(name, device), "float32"

    def to(self, target):
        import torch

        self.model.to(target)
        if isinstance(target, torch.dtype):
            self.head = self.model.dtype
        else:
            self.device = str(target)
        return self

    def predict(self, pairs, batch_size, show_progress_bar=None):
        assert show_progress_bar is False, "sift must not draw progress bars (the server's log is no terminal)"
        if self.model.dtype != self.head:
            raise RuntimeError(f"expected mat1 and mat2 to have the same dtype, but got: {self.model.dtype} != float")
        return np.array([len(set(words(q)) & set(words(d))) / (1 + len(set(words(q)))) for q, d in pairs])


GiB = 2**30


class FakeGpuMemory:
    """Stands in for torch.cuda's memory API: 20 of 24 GiB free, another program holds the rest.

    The loaded model takes 1 GiB, and a batch of n texts at full length needs 0.1 + 0.05 * n GiB on top.
    """

    weights, fixed, per_text = 1 * GiB, 0.1 * GiB, 0.05 * GiB
    oom_from = None  # batch size from which to fail anyway, like fragmented memory

    def __init__(self):
        self.free, self.total, self.budgets, self.batches = 20 * GiB, 24 * GiB, [], []
        self.used = None  # set it to make NVML disagree with CUDA, as it does on Windows
        self.peak, self.emptied = self.weights, 0

    def mem_get_info(self, device=None):
        return self.free, self.total

    def device_memory_used(self, device=None):
        """What NVML (and nvidia-smi) report, other programs included."""
        return self.total - self.free if self.used is None else self.used

    def set_per_process_memory_fraction(self, fraction, device=None):
        if not isinstance(device, int):  # like PyTorch, which rejects torch.device("cuda") without an index
            raise ValueError(f"Expected a torch.device with a specified index or an integer, but got:{device}")
        self.budgets.append((fraction, device))

    def current_device(self):
        return 0

    def memory_allocated(self, device=None):
        return self.weights

    def max_memory_allocated(self, device=None):
        return self.peak

    def reset_peak_memory_stats(self, device=None):
        self.peak = self.weights

    def empty_cache(self):
        self.emptied += 1

    def run(self, n):
        """A batch of n texts on the GPU; out of memory past the budget, like PyTorch."""
        import torch

        need = self.weights + self.fixed + n * self.per_text
        limit = self.budgets[-1][0] * self.total if self.budgets else self.free
        self.batches.append(n)
        if need > limit or (self.oom_from and n >= self.oom_from):
            raise torch.cuda.OutOfMemoryError(f"CUDA out of memory. Tried to allocate {need / GiB:.2f} GiB")
        self.peak = max(self.peak, need)


@pytest.fixture
def gpu_memory(monkeypatch):
    fake = FakeGpuMemory()
    monkeypatch.setattr(FakeSentenceTransformer, "gpu_memory", fake)
    return fake


@pytest.fixture(autouse=True)
def close_databases(monkeypatch):
    """Close every database a test opens; one left to the garbage collector is a ResourceWarning (Python 3.13+)."""
    opened, connect = [], store.connect

    def tracked(*args, **kwargs):
        opened.append(db := connect(*args, **kwargs))
        return db

    monkeypatch.setattr(store, "connect", tracked)
    yield
    for db in opened:
        db.close()


class FakeClock:
    """Stands in for sift's clocks: it stands still until a test moves it, so no test depends on real time."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    from sift import console

    fake = FakeClock()
    monkeypatch.setattr(console, "perf_counter", fake)  # the progress line's
    return fake


class PretendGpu(Accelerator):
    """The CPU, managed as if it were a GPU: tensors stay on the CPU, but move "to the GPU" and back as on one, so the
    bookkeeping is tested without a GPU."""

    gpu = True


@pytest.fixture
def make_local(monkeypatch, gpu_memory, clock):
    """Builds the models of this machine around fakes: Sentence Transformers, GPU memory and clock. Everything else,
    the accelerator's decisions included, runs for real."""
    import torch

    from sift.local import Local

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer, module.CrossEncoder = FakeSentenceTransformer, FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)  # a GPU of this decade

    def make(device="cpu", accelerator=Accelerator):
        return Local(accelerator(device, memory=gpu_memory, clock=clock))

    return make


@pytest.fixture
def configure(monkeypatch):
    """Change sift's settings for one test, as if the environment said so: configure(remote=..., home=...)."""
    from dataclasses import replace

    from sift import settings

    def change(**changes):
        changed = replace(settings.current(), **changes)
        monkeypatch.setattr(settings, "current", lambda: changed)

    return change


@pytest.fixture
def fake_models(make_local, configure, monkeypatch):
    """The models sift uses by default: those of this machine, on the CPU, with fakes inside."""
    from sift import models

    local = make_local()
    monkeypatch.setattr(models, "_local", lambda: local)
    configure(remote=None)  # this machine embeds, whatever SIFT_EMBED_REMOTE says
    models._remote.cache_clear()
    yield local
    models._remote.cache_clear()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")  # the same bytes on every OS


def nobody_listens():
    """A remote no connection can reach. Port 0 fails at once; a closed port takes 2 s to refuse on Windows."""
    return "https://s3cret.pin@127.0.0.1:0"


def index_contents(db):
    """Everything indexing produces, independent of row ids."""
    return sorted(
        db.execute("""SELECT d.path, d.hash, d.title, d.meta, c.seq, c.line_start, c.line_end, t.text
                      FROM docs d JOIN chunks c ON c.doc = d.id JOIN chunk_text t ON t.id = c.id""")
    )


@pytest.fixture
def pools(monkeypatch):
    """Records every worker process pool that indexing starts."""
    created = []

    class RecordingPool(indexing.ProcessPoolExecutor):
        def __init__(self, max_workers, mp_context=None):
            created.append((max_workers, mp_context and mp_context.get_start_method()))
            super().__init__(max_workers, mp_context=mp_context)

    monkeypatch.setattr(indexing, "ProcessPoolExecutor", RecordingPool)
    return created


@pytest.fixture
def notes(tmp_path):
    """A small folder of notes."""
    root = tmp_path / "notes"
    write(
        root / "robots" / "brush.md",
        '---\ntitle: "Brush stroke planning"\ncategories: ["cs.RO", "cs.GR"]\nyear: 2024\n---\n\n'
        "# Brush stroke planning\n\nA painting robot plans brush strokes with Kubelka-Munk pigment mixing.\n",
    )
    write(root / "rl.md", "# Reward shaping\n\nReward shaping speeds up reinforcement learning.\n")
    write(root / "plain.txt", "Colour theory notes without any heading.\n")
    write(root / "ignored.py", "print('not indexed')\n")
    write(root / ".hidden" / "secret.md", "# Hidden\n")
    write(root / "node_modules" / "dep.md", "# Dependency\n")
    return root


@pytest.fixture
def home(tmp_path, configure):
    """SIFT_HOME for code that connects on its own (CLI, server)."""
    path = tmp_path / "home"
    configure(home=path)
    return path


@pytest.fixture
def corpus(notes, home):
    """An index of `notes`, keyword search ready, not embedded."""
    db = store.connect(home)
    store.add_collection(db, notes, "notes")
    indexing.index(db)
    return db, notes


@pytest.fixture
def embedded(corpus, fake_models):
    from sift import embed

    embed.embed_pending(corpus[0])
    return corpus
