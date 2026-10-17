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


def fake_vector(text):
    """Deterministic bag-of-words embedding: texts sharing words point in similar directions."""
    v = np.zeros(DIM, np.float32)
    for w in words(text):
        v[zlib.crc32(w.encode()) % DIM] += 1
    return v / (np.linalg.norm(v) or 1)


class FakeSentenceTransformer:
    max_batch = 10**6  # batches above this raise CUDA out-of-memory

    def __init__(self, name, device=None):
        self.name, self.device, self.max_seq_length, self.halved = name, device, None, False

    def half(self):
        self.halved = True
        return self

    def encode_document(self, texts, batch_size, normalize_embeddings, convert_to_numpy):
        import torch

        if batch_size > self.max_batch:
            raise torch.cuda.OutOfMemoryError("fake out of memory")
        return np.stack([fake_vector(t) for t in texts])

    encode_query = encode_document


class FakeCrossEncoder:
    def __init__(self, name, device=None, max_length=None):
        self.name, self.model = name, FakeSentenceTransformer(name)

    def predict(self, pairs, batch_size):
        return np.array([len(set(words(q)) & set(words(d))) / (1 + len(set(words(q)))) for q, d in pairs])


@pytest.fixture
def fake_models(monkeypatch):
    """Replace Sentence Transformers with the fakes above; everything else in sift runs for real."""
    from sift import embed

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer, module.CrossEncoder = FakeSentenceTransformer, FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(embed, "DEVICE", "cpu")
    monkeypatch.setattr(embed, "_batch_size", 128)
    embed.embedder.cache_clear()
    embed.reranker.cache_clear()
    yield embed
    embed.embedder.cache_clear()
    embed.reranker.cache_clear()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
