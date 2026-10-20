import functools
import re
import sys
import types
import zlib

import numpy as np
import pytest

from sift import store

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
        self.name, self.device, self.max_seq_length, self.halved = name, device, None, False

    def half(self):
        self.halved = True
        return self

    def to(self, device):
        self.device = str(device)
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
    def __init__(self, name, device=None, max_length=None):
        self.name, self.device, self.model = name, device, FakeSentenceTransformer(name, device)

    def to(self, device):
        self.device = str(device)
        self.model.to(device)
        return self

    def predict(self, pairs, batch_size, show_progress_bar=None):
        assert show_progress_bar is False, "sift must not draw progress bars (the server's log is no terminal)"
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
        self.peak = self.weights

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
    import torch

    fake = FakeGpuMemory()
    for name in (
        "mem_get_info",
        "set_per_process_memory_fraction",
        "memory_allocated",
        "max_memory_allocated",
        "reset_peak_memory_stats",
        "current_device",
        "device_memory_used",
    ):
        monkeypatch.setattr(torch.cuda, name, getattr(fake, name))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
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
    from sift import embed

    fake = FakeClock()
    monkeypatch.setattr(embed, "monotonic", fake)
    monkeypatch.setattr(embed, "perf_counter", fake)
    return fake


@pytest.fixture
def fake_models(monkeypatch, gpu_memory, clock):
    """Replace Sentence Transformers, GPU memory queries and clocks with fakes; everything else runs for real."""
    from sift import embed

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer, module.CrossEncoder = FakeSentenceTransformer, FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(embed, "DEVICE", "cpu")
    monkeypatch.setattr(embed, "_batch_size", 128)
    monkeypatch.setattr(embed, "_resident", {})  # nothing on the GPU yet
    monkeypatch.setattr(embed, "_released", False)
    caches = (embed.embedder, embed.reranker, embed.limit_gpu_memory, embed._remote_info)
    for cached in caches:
        cached.cache_clear()
    yield embed
    for cached in caches:
        cached.cache_clear()


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

    class RecordingPool(store.ProcessPoolExecutor):
        def __init__(self, max_workers, mp_context=None):
            created.append((max_workers, mp_context and mp_context.get_start_method()))
            super().__init__(max_workers, mp_context=mp_context)

    monkeypatch.setattr(store, "ProcessPoolExecutor", RecordingPool)
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
def home(tmp_path, monkeypatch):
    """SIFT_HOME for code that connects on its own (CLI, server)."""
    path = tmp_path / "home"
    monkeypatch.setattr(store, "HOME", path)
    return path


@pytest.fixture
def corpus(notes, home):
    """An index of `notes`, keyword search ready, not embedded."""
    db = store.connect(home)
    store.add_collection(db, notes, "notes")
    store.index(db)
    return db, notes


@pytest.fixture
def embedded(corpus, fake_models):
    fake_models.embed_pending(corpus[0])
    return corpus
