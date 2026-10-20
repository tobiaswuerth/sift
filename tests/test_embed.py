import logging
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from sift import store

from .conftest import FakeSentenceTransformer, GiB, fake_vector, write


def vector_count(db):
    return db.execute("SELECT count(*) FROM vectors").fetchone()[0]


def models_in_a_fresh_process(**env):
    """The models sift.embed picks when imported with these environment variables, and without sift's own."""
    env = {k: v for k, v in os.environ.items() if k not in ("SIFT_EMBED_MODEL", "SIFT_RERANK_MODEL")} | env
    code = "from sift import embed; print(embed.EMBED_MODEL, embed.RERANK_MODEL)"
    run = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    return tuple(run.stdout.split())


def test_default_models():
    assert models_in_a_fresh_process() == ("google/embeddinggemma-2", "cross-encoder/ettin-reranker-150m-v1")
    assert models_in_a_fresh_process(SIFT_EMBED_MODEL="a/embedder", SIFT_RERANK_MODEL="b/reranker") == (
        "a/embedder",
        "b/reranker",
    )


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
    assert vecs.dtype == np.float16
    assert vecs.shape == (2, 64)
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


def test_embed_pending_reports_progress_within_a_slab(corpus, fake_models, monkeypatch, capsys):
    db, root = corpus
    for i in range(5):
        write(root / f"extra{i}.md", f"# Extra {i}\n\n" + "word " * (10 + 50 * i) + "\n")
    store.index(db)
    monkeypatch.setattr(fake_models, "_batch_size", 1)
    monkeypatch.setattr(fake_models, "PROGRESS_BATCHES", 2)  # report after every 2 batches, long before the slab ends
    fake_models.embed_pending(db)
    err = capsys.readouterr().err
    assert all(f"embedded {n}/8 chunks" in err for n in (2, 4, 6, 8))
    # sorting by length for the progress steps must not mix up which vector belongs to which chunk
    rows = db.execute("""SELECT c.seq, t.title, t.text, v.vec FROM chunks c JOIN chunk_text t ON t.id = c.id
                         JOIN vectors v ON v.chunk = c.id""").fetchall()
    assert len(rows) == 8
    for seq, title, text, vec in rows:
        expected = fake_vector(fake_models._text(seq, title, text)).astype(np.float16)
        assert np.frombuffer(vec, np.float16).tolist() == expected.tolist()


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
    assert later
    assert all("more" in t or "word" in t for t in later)


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
    assert gpu_memory.budgets == [(pytest.approx((20 - 2.4) / 24), 0)]  # free minus 10% of the GPU
    assert "using at most 17.6 of 24.0 GB of GPU memory" in capsys.readouterr().err


def test_gpu_memory_budget_counts_memory_other_programs_hold(fake_models, gpu_memory, monkeypatch):
    # On Windows (WDDM), CUDA reports memory held by other programs as free, since the driver could page it out.
    # NVML, which nvidia-smi uses, does not; the smaller of the two decides.
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    gpu_memory.free, gpu_memory.used = 22.5 * GiB, 13.5 * GiB  # as measured on an RTX 4090 under Windows
    fake_models.embedder()
    assert gpu_memory.budgets == [(pytest.approx((24 - 13.5 - 2.4) / 24), 0)]


def test_gpu_memory_budget_without_nvml_uses_cuda(fake_models, gpu_memory, monkeypatch):
    import torch

    def no_nvml(device=None):
        raise ModuleNotFoundError("nvidia-ml-py does not seem to be installed")

    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "device_memory_used", no_nvml)
    fake_models.embedder()
    assert gpu_memory.budgets == [(pytest.approx((20 - 2.4) / 24), 0)]


def test_gpu_memory_headroom_is_at_least_one_gb(fake_models, gpu_memory, monkeypatch):
    monkeypatch.setattr(fake_models, "DEVICE", "cuda:1")
    gpu_memory.free, gpu_memory.total = 6 * GiB, 8 * GiB
    fake_models.embedder()
    assert gpu_memory.budgets == [(pytest.approx(5 / 8), 1)]


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
    monkeypatch.setattr(gpu_memory, "oom_from", 16)  # fails earlier than the linear model predicts
    size = fake_models.calibrate_batch_size()
    assert gpu_memory.batches == [1, 2, 4, 8, 16]
    assert size == 15  # never a size that already failed


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


def test_embed_pending_calibrates_on_the_longest_chunk(corpus, fake_models, monkeypatch):
    db, root = corpus
    write(root / "long.md", "# Long\n" + "word " * 1000 + "\n\n" + "more " * 1000 + "\n")
    store.index(db)
    samples, embedded = [], []
    real_encode = fake_models.encode
    monkeypatch.setattr(fake_models, "calibrate_batch_size", lambda sample=None: samples.append(sample) or 128)
    monkeypatch.setattr(fake_models, "encode", lambda texts, query=False: embedded.extend(texts) or real_encode(texts))
    fake_models.embed_pending(db)
    assert samples == [max(embedded, key=len)]  # no batch in this run can be longer


def test_calibration_uses_the_sample(fake_models, gpu_memory, monkeypatch):
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    seen, real = [], FakeSentenceTransformer.encode_document

    def recording(self, texts, **kwargs):
        seen.extend(texts)
        return real(self, texts, **kwargs)

    monkeypatch.setattr(FakeSentenceTransformer, "encode_document", recording)
    fake_models.calibrate_batch_size("a sample text")
    assert set(seen) == {"a sample text"}


def test_harmless_torch_windows_note_is_silenced(fake_models):
    redirects = logging.getLogger("torch.distributed.elastic.multiprocessing.redirects")
    assert redirects.getEffectiveLevel() >= logging.ERROR


@pytest.fixture
def nvml(monkeypatch):
    """A fake NVML (nvidia-ml-py) that reports one NVIDIA GPU."""
    module = types.ModuleType("pynvml")
    module.gpus = 1
    module.nvmlInit = lambda: None
    module.nvmlDeviceGetCount = lambda: module.gpus
    monkeypatch.setitem(sys.modules, "pynvml", module)
    return module


@pytest.mark.parametrize(
    "cuda_build, hint",
    [
        (None, "uv tool install --reinstall --torch-backend auto sift-rag"),  # uv environments have no pip
        (None, "--index-url https://download.pytorch.org/whl/cu128"),
        ("12.8", "update the NVIDIA driver"),
    ],
)
def test_explains_an_unused_nvidia_gpu(fake_models, nvml, monkeypatch, capsys, cuda_build, hint):
    import torch

    monkeypatch.setattr(torch.version, "cuda", cuda_build)
    fake_models.embedder()
    fake_models.reranker()
    err = capsys.readouterr().err
    assert err.count("found an NVIDIA GPU") == 1  # once, not per model
    assert hint in err


def test_no_gpu_hint_without_an_nvidia_gpu(fake_models, nvml, monkeypatch, capsys):
    nvml.gpus = 0
    fake_models.embedder()
    assert "NVIDIA" not in capsys.readouterr().err


def test_no_gpu_hint_without_nvml(fake_models, nvml, monkeypatch, capsys):
    def broken():
        raise OSError("NVML Shared Library Not Found")

    monkeypatch.setattr(nvml, "nvmlInit", broken)
    fake_models.embedder()
    assert "NVIDIA" not in capsys.readouterr().err


def test_no_gpu_hint_on_apple_gpus(fake_models, nvml, monkeypatch, capsys):
    monkeypatch.setattr(fake_models, "DEVICE", "mps")
    fake_models.embedder()
    assert "NVIDIA" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "platform, command",
    [
        # Windows CUDA wheels bundle their CUDA libraries; Linux ones depend on nvidia-* packages
        (
            "win32",
            "pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu128",
        ),
        ("linux", "pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128"),
    ],
)
def test_cuda_pytorch_command_fits_the_platform(fake_models, platform, command):
    assert fake_models.cuda_pytorch_command(platform) == command


# --- models wait in RAM and use the GPU only while busy ------------------------


def test_models_wait_in_ram_until_used(fake_models, monkeypatch):
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    embedder, reranker = fake_models.embedder(), fake_models.reranker()
    assert (embedder.device, reranker.device) == ("cpu", "cpu")
    assert embedder.halved  # fp16 already, so moving to the GPU is quick
    fake_models.encode(["brush strokes"])
    assert (embedder.device, reranker.device) == ("cuda", "cpu")
    fake_models.rerank("brush", ["brush strokes"])
    assert reranker.device == "cuda"


def test_idle_models_leave_the_gpu(fake_models, clock, gpu_memory, monkeypatch, capsys):
    import torch

    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    fake_models.encode(["brush strokes"])
    fake_models.rerank("brush", ["brush strokes"])
    clock.advance(fake_models.IDLE_SECONDS - 0.5)
    assert not fake_models.release_idle()
    assert fake_models.embedder().device == "cuda"
    clock.advance(0.5)  # exactly IDLE_SECONDS since the last use
    assert fake_models.release_idle()
    assert (fake_models.embedder().device, fake_models.reranker().device) == ("cpu", "cpu")
    assert emptied  # the memory goes back to the GPU, not just to PyTorch's cache
    assert "moved off the GPU" in capsys.readouterr().err
    clock.advance(10 * fake_models.IDLE_SECONDS)
    assert not fake_models.release_idle()  # nothing left to release


def test_every_use_restarts_the_idle_time(fake_models, clock, monkeypatch):
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    fake_models.encode(["brush"])
    clock.advance(fake_models.IDLE_SECONDS - 1)
    fake_models.encode(["brush"])  # already on the GPU, still counts as use
    clock.advance(fake_models.IDLE_SECONDS - 1)
    assert not fake_models.release_idle()
    clock.advance(1)
    assert fake_models.release_idle()


def test_gpu_budget_is_refreshed_when_coming_back(fake_models, clock, gpu_memory, monkeypatch):
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    fake_models.encode(["brush"])
    assert len(gpu_memory.budgets) == 1  # set once while loading
    clock.advance(fake_models.IDLE_SECONDS)
    assert fake_models.release_idle()
    gpu_memory.free = 10 * GiB  # another program took more memory meanwhile
    fake_models.encode(["brush"])
    assert gpu_memory.budgets[-1] == (pytest.approx((10 - 2.4) / 24), 0)


def test_nothing_moves_on_the_cpu(fake_models, clock):
    fake_models.encode(["brush"])
    assert fake_models.embedder().device == "cpu"
    clock.advance(10 * fake_models.IDLE_SECONDS)
    assert not fake_models.release_idle()


def test_idle_release_on_the_limit_despite_rounding(fake_models, clock, monkeypatch):
    # (a + 300) - a is 299.9999999999982 for this clock reading: comparing the idle time with the limit missed
    # the release by a rounding error, and failed two CI jobs.
    clock.now = 16165.871537361332
    monkeypatch.setattr(fake_models, "gpu", lambda: True)
    released = []
    fake_models.acquire("thing", lambda: None, lambda: released.append(True))
    clock.advance(fake_models.IDLE_SECONDS)
    assert fake_models.release_idle()
    assert released == [True]


def test_embed_progress_survives_a_clock_that_stands_still(corpus, fake_models, capsys):
    # Coarse timers can return the same reading twice; the progress rate must not divide by zero.
    fake_models.embed_pending(corpus[0])
    assert "embedded 3/3 chunks" in capsys.readouterr().err


def test_calibration_when_memory_does_not_grow_with_the_batch(fake_models, gpu_memory, monkeypatch):
    # Tiny texts can leave the measured peak unchanged, and the estimate divided by zero.
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    monkeypatch.setattr(gpu_memory, "per_text", 0)
    assert fake_models.calibrate_batch_size() == fake_models.MAX_BATCH


def test_coming_back_to_a_busy_gpu_uses_the_cpu(fake_models, clock, gpu_memory, monkeypatch, capsys):
    # Another program took the GPU memory while sift was idle. Moving back anyway could crash that program.
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    fake_models.encode(["brush"])
    clock.advance(fake_models.IDLE_SECONDS)
    assert fake_models.release_idle()
    gpu_memory.free = 2 * GiB
    assert fake_models.encode(["brush"]).shape == (1, 64)
    assert fake_models.rerank("brush", ["brush strokes"]).shape == (1,)
    embedder, reranker = fake_models.embedder(), fake_models.reranker()
    assert (fake_models.DEVICE, embedder.device, reranker.device) == ("cpu", "cpu", "cpu")
    assert not embedder.halved  # float32 again: CPUs are slow at fp16
    assert not reranker.model.halved
    assert len(gpu_memory.budgets) == 1  # no budget for memory that isn't there
    assert "using the CPU" in capsys.readouterr().err


def test_embed_pending_skips_chunks_without_text(embedded, fake_models, capsys):
    # A chunk without a document can only come from an index changed meanwhile; there's nothing to embed.
    db, _ = embedded
    capsys.readouterr()
    db.execute("INSERT INTO chunks(doc, seq, line_start, line_end, pos, len) VALUES(999, 9, 1, 1, 0, 0)")
    fake_models.embed_pending(db)  # raised "max() arg is an empty sequence"
    assert "nothing to embed" in capsys.readouterr().err


def test_no_calibration_after_giving_up_the_gpu(fake_models, clock, gpu_memory, monkeypatch):
    monkeypatch.setattr(fake_models, "DEVICE", "cuda")
    fake_models.encode(["brush"])
    clock.advance(fake_models.IDLE_SECONDS)
    fake_models.release_idle()
    gpu_memory.free = 2 * GiB
    measured = len(gpu_memory.batches)
    assert fake_models.calibrate_batch_size() == 128
    assert len(gpu_memory.batches) == measured  # nothing ran on the GPU
