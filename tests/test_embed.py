import logging

import numpy as np
import pytest

from sift import store

from .conftest import FakeSentenceTransformer, GiB, write


def vector_count(db):
    return db.execute("SELECT count(*) FROM vectors").fetchone()[0]


def test_models_load_with_limits(fake_models):
    model = fake_models.embedder()
    assert (model.name, model.device, model.max_seq_length, model.halved) == (
        fake_models.EMBED_MODEL,
        "cpu",
        fake_models.MAX_TOKENS,
        False,
    )
    assert fake_models.reranker().name == fake_models.RERANK_MODEL
    assert fake_models.embedder() is model  # loaded once


@pytest.mark.parametrize("cuda, mps, device", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_device_prefers_cuda_then_mps_then_cpu(fake_models, monkeypatch, cuda, mps, device):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert fake_models._device() == device


@pytest.mark.parametrize("device, fp16", [("cuda", True), ("cuda:1", True), ("mps", True), ("cpu", False)])
def test_models_use_fp16_on_gpus(fake_models, monkeypatch, device, fp16):
    monkeypatch.setattr(fake_models, "DEVICE", device)
    assert fake_models.fp16() == fp16
    assert fake_models.embedder().halved == fp16
    assert fake_models.reranker().model.halved == fp16


def test_encode_returns_normalised_fp16(fake_models):
    vecs = fake_models.encode(["brush strokes", "reward shaping"])
    assert vecs.dtype == np.float16 and vecs.shape == (2, 64)
    assert np.allclose(np.linalg.norm(vecs.astype(np.float32), axis=1), 1, atol=1e-2)
    assert fake_models.encode(["brush"], query=True).shape == (1, 64)


def test_rerank_scores_pairs(fake_models):
    scores = fake_models.rerank("brush strokes", ["brush strokes here", "nothing related"])
    assert scores[0] > scores[1]


@pytest.mark.parametrize("gpu", ["cuda", "mps"])
def test_out_of_memory_halves_batch_size(fake_models, monkeypatch, capsys, gpu):
    import torch

    emptied = []
    monkeypatch.setattr(getattr(torch, gpu), "empty_cache", lambda: emptied.append(gpu))
    monkeypatch.setattr(FakeSentenceTransformer, "gpu", gpu)
    monkeypatch.setattr(FakeSentenceTransformer, "max_batch", 20)
    fake_models.encode(["a", "b"])
    assert fake_models._batch_size == 16
    assert emptied == [gpu] * 3  # 128, 64 and 32 failed
    assert "batch size now 16" in capsys.readouterr().err


@pytest.mark.parametrize("gpu", ["cuda", "mps"])
def test_out_of_memory_at_batch_size_one_raises(fake_models, monkeypatch, gpu):
    import torch

    monkeypatch.setattr(getattr(torch, gpu), "empty_cache", lambda: None)
    monkeypatch.setattr(FakeSentenceTransformer, "gpu", gpu)
    monkeypatch.setattr(FakeSentenceTransformer, "max_batch", 0)
    with pytest.raises(RuntimeError, match="out of memory"):
        fake_models.encode(["a"])
    assert fake_models._batch_size == 1


def test_other_runtime_errors_are_not_retried(fake_models, monkeypatch):
    def broken(self, texts, **kwargs):
        raise RuntimeError("Expected all tensors to be on the same device")

    monkeypatch.setattr(FakeSentenceTransformer, "encode_document", broken)
    with pytest.raises(RuntimeError, match="same device"):
        fake_models.encode(["a"])
    assert fake_models._batch_size == 128


def test_embed_pending(corpus, fake_models, capsys):
    db, _ = corpus
    fake_models.embed_pending(db)
    assert vector_count(db) == 3
    assert store.get_meta(db, "model") == fake_models.EMBED_MODEL
    assert store.get_meta(db, "dim") == "64"
    assert "embedded 3 chunks" in capsys.readouterr().err
    fake_models.embed_pending(db)
    assert "nothing to embed" in capsys.readouterr().err


def test_embed_pending_in_slabs_with_title_on_later_chunks(corpus, fake_models, monkeypatch):
    db, root = corpus
    write(root / "long.md", "# Long\n" + "word " * 1000 + "\n" * 2 + "more " * 1000 + "\n")
    store.index(db)
    seen = []
    real_encode = fake_models.encode
    monkeypatch.setattr(fake_models, "encode", lambda texts, query=False: seen.extend(texts) or real_encode(texts))
    monkeypatch.setattr(fake_models, "SLAB", 2)
    fake_models.embed_pending(db)
    assert vector_count(db) == db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    later = [t for t in seen if t.startswith("Long\n\n")]
    assert later and all("more" in t or "word" in t for t in later)


def test_embed_pending_survives_concurrent_writer(corpus, fake_models, monkeypatch):
    db, _ = corpus
    real_encode = fake_models.encode

    def racing_encode(texts, query=False):  # another process embeds the first chunk meanwhile
        vecs = real_encode(texts)
        db.execute("INSERT INTO vectors(chunk, vec) SELECT min(id), x'00' FROM chunks")
        return vecs

    monkeypatch.setattr(fake_models, "encode", racing_encode)
    fake_models.embed_pending(db)
    assert vector_count(db) == 3


def test_embed_refuses_mixing_models(embedded, fake_models):
    db, _ = embedded
    store.set_meta(db, "model", "another/model")
    with pytest.raises(SystemExit, match="--reset"):
        fake_models.embed_pending(db)
    fake_models.embed_pending(db, reset=True)
    assert store.get_meta(db, "model") == fake_models.EMBED_MODEL
    assert vector_count(db) == 3


def test_gpu_memory_budget_leaves_room_for_other_programs(fake_models, gpu_memory, monkeypatch, capsys):
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    fake_models.embedder()
    fake_models.reranker()  # loading a second model sets no second budget
    assert gpu_memory.budgets == [(pytest.approx((20 - 2.4) / 24), "cuda")]  # free minus 10% of the GPU
    assert "using at most 17.6 of 24.0 GB of GPU memory" in capsys.readouterr().err


def test_gpu_memory_headroom_is_at_least_one_gb(fake_models, gpu_memory, monkeypatch):
    monkeypatch.setattr(fake_models, "DEVICE", "cuda:1")
    gpu_memory.free, gpu_memory.total = 6 * GiB, 8 * GiB
    fake_models.embedder()
    assert gpu_memory.budgets == [(pytest.approx(5 / 8), "cuda:1")]


def test_too_little_free_gpu_memory_falls_back_to_cpu(fake_models, gpu_memory, monkeypatch, capsys):
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    gpu_memory.free = 2 * GiB  # less than the 2 GiB budget the models need, after headroom
    model = fake_models.embedder()
    assert (model.device, fake_models.DEVICE, model.halved) == ("cpu", "cpu", False)
    assert gpu_memory.budgets == []
    assert "only 2.0 GB of GPU memory is free; using the CPU" in capsys.readouterr().err


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_no_gpu_memory_budget_off_cuda(fake_models, gpu_memory, monkeypatch, device):
    monkeypatch.setattr(fake_models, "DEVICE", device)
    fake_models.embedder()
    fake_models.reranker()
    assert gpu_memory.budgets == []


def test_batch_size_is_calibrated_to_fill_the_budget(fake_models, gpu_memory, monkeypatch, capsys):
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    size = fake_models.calibrate_batch_size()
    # budget 17.6 GiB; 90% of it minus 1 GiB of weights leaves 14.84 GiB for 0.1 + 0.05 per text
    assert size == 1 + int((0.9 * 17.6 - 1 - 0.15) / 0.05) == 294
    assert fake_models._batch_size == size
    assert gpu_memory.batches == [1, 2, 4, 8, 16, 32, 64, 128, 256]  # doubled until half the budget was in use
    assert "batch size 294" in capsys.readouterr().err
    fake_models.encode(["x"] * 600)  # the calibrated size fits: no out-of-memory, no halving
    assert fake_models._batch_size == 294


def test_calibration_stops_at_the_first_out_of_memory(fake_models, gpu_memory, monkeypatch):
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    monkeypatch.setattr(gpu_memory, "per_text", 1 * GiB)  # 16 texts are already too many
    size = fake_models.calibrate_batch_size()
    assert gpu_memory.batches[-1] == 16
    assert 1 <= size <= 15


def test_calibration_is_capped(fake_models, gpu_memory, monkeypatch):
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    monkeypatch.setattr(gpu_memory, "per_text", 0.001 * GiB)
    assert fake_models.calibrate_batch_size() == fake_models.MAX_BATCH


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_no_calibration_off_cuda(fake_models, gpu_memory, monkeypatch, device):
    monkeypatch.setattr(fake_models, "DEVICE", device)
    assert fake_models.calibrate_batch_size() == 128
    assert gpu_memory.batches == []


def test_embed_pending_calibrates_first(corpus, fake_models, gpu_memory, monkeypatch):
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    fake_models.embed_pending(corpus[0])
    assert gpu_memory.batches[:3] == [1, 2, 4]  # calibration, then the 3 chunks in one batch
    assert gpu_memory.batches[-1] == 3


def test_harmless_torch_windows_note_is_silenced(fake_models):
    redirects = logging.getLogger("torch.distributed.elastic.multiprocessing.redirects")
    assert redirects.getEffectiveLevel() >= logging.ERROR
