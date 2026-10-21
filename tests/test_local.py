"""The models of this machine: an embedding model and a reranker, loaded once per precision, on the GPU while in use.
Sentence Transformers, GPU memory and the clock are fakes."""

import numpy as np
import pytest

from sift import accelerator, local
from sift.vectors import InRam, OnGpu

from .conftest import FakeCrossEncoder, FakeSentenceTransformer, GiB


def test_models_load_once_with_limits(fake_models):
    model = fake_models.embedder()
    assert (model.name, model.device, model.max_seq_length, model.dtype) == (
        "google/embeddinggemma-2",
        "cpu",
        local.MAX_TOKENS,
        "float32",
    )
    assert fake_models.reranker().name == "cross-encoder/ettin-reranker-150m-v1"
    assert fake_models.embedder() is model
    assert (fake_models.name, fake_models.reranker_name) == ("google/embeddinggemma-2", fake_models.reranker().name)


@pytest.mark.parametrize("device, dtype", [("cuda", "bfloat16"), ("mps", "float32"), ("cpu", "float32")])
def test_models_compute_in_the_devices_half_precision(make_local, device, dtype):
    models = make_local(device)
    assert models.embedder().dtype == models.reranker().model.dtype == dtype
    assert models.rerank("brush", ["brush strokes"]).shape == (1,)  # the reranker's scoring head converted too


def test_encode_returns_normalised_fp16(fake_models):
    vecs = fake_models.encode(["brush strokes", "reward shaping"])
    assert vecs.dtype == np.float16
    assert vecs.shape == (2, 64)
    assert np.allclose(np.linalg.norm(vecs.astype(np.float32), axis=1), 1, atol=1e-2)
    assert fake_models.encode(["brush"], query=True).shape == (1, 64)


def test_rerank_scores_pairs(fake_models):
    scores = fake_models.rerank("brush strokes", ["brush strokes here", "nothing related"])
    assert scores[0] > scores[1]


def test_rerank_gives_raw_scores_whatever_the_model_applies(fake_models, monkeypatch):
    # A sigmoid saturates: a reranker that applies one would score every good match 0.9999.
    import torch

    monkeypatch.setattr(FakeCrossEncoder, "activation", torch.nn.Sigmoid())
    assert fake_models.rerank("brush strokes", ["brush strokes"]).tolist() == pytest.approx([2 / 3])


def test_loading_the_models_draws_no_progress_bars(fake_models):
    # Transformers drew a "Loading weights" bar for each model, into the log of `sift serve`, garbled on Windows.
    from transformers.utils import logging

    logging.enable_progress_bar()
    fake_models.embedder()
    fake_models.reranker()
    assert not logging.is_progress_bar_enabled()


def test_a_reranker_that_reads_fewer_tokens_keeps_its_limit(fake_models, monkeypatch):
    # sift asked every reranker for 1024 tokens; one that reads at most 512 (MiniLM) failed on every long passage:
    # "The size of tensor a (972) must match the size of tensor b (512)".
    monkeypatch.setattr(FakeCrossEncoder, "limit", 512)
    assert fake_models.rerank("brush strokes", ["brush strokes"]).shape == (1,)
    assert fake_models.reranker().max_seq_length == 512


def test_a_reranker_that_reads_more_tokens_is_capped(fake_models):
    assert fake_models.reranker().max_seq_length == local.MAX_TOKENS  # longer inputs cost time and memory


def test_where_and_how_much(make_local):
    models = make_local("cuda")
    assert (models.address, models.batch) == ("cuda", 128)
    models.accelerator.batch_size = 64
    assert models.batch == 64


def test_out_of_memory_while_encoding_halves_the_batch(make_local, monkeypatch):
    monkeypatch.setattr(FakeSentenceTransformer, "max_batch", 20)
    models = make_local("cuda")
    assert models.encode(["a", "b"]).shape == (2, 64)
    assert models.batch == 16


def test_models_wait_in_ram_until_used(make_local):
    models = make_local("cuda")
    embedder, reranker = models.embedder(), models.reranker()
    assert (embedder.device, reranker.device) == ("cpu", "cpu")  # in half precision already: moving is quick
    models.encode(["brush strokes"])
    assert (embedder.device, reranker.device) == ("cuda", "cpu")
    models.rerank("brush", ["brush strokes"])
    assert reranker.device == "cuda"


def test_idle_models_leave_the_gpu(make_local, clock):
    models = make_local("cuda")
    models.encode(["brush strokes"])
    models.rerank("brush", ["brush strokes"])
    clock.advance(accelerator.IDLE_SECONDS)
    assert models.release_idle()
    assert (models.embedder().device, models.reranker().device) == ("cpu", "cpu")


def test_back_on_a_busy_gpu_the_models_reload_for_the_cpu(make_local, gpu_memory, clock):
    models = make_local("cuda")
    models.encode(["brush"])
    clock.advance(accelerator.IDLE_SECONDS)
    assert models.release_idle()
    gpu_memory.free = 2 * GiB  # another program took the GPU meanwhile
    assert models.encode(["brush"]).shape == (1, 64)
    assert models.rerank("brush", ["brush strokes"]).shape == (1,)
    assert (models.address, models.embedder().device, models.reranker().device) == ("cpu", "cpu", "cpu")
    assert models.embedder().dtype == models.reranker().model.dtype == "float32"  # CPUs are slow at half precision


def test_prepare_calibrates_on_the_longest_text(make_local, gpu_memory, monkeypatch):
    seen, real = [], FakeSentenceTransformer.encode_document

    def recording(self, texts, **kwargs):
        seen.extend(texts)
        return real(self, texts, **kwargs)

    monkeypatch.setattr(FakeSentenceTransformer, "encode_document", recording)
    models = make_local("cuda")
    models.prepare(lambda: "a sample text")
    assert set(seen) == {"a sample text"}
    assert models.batch == 294  # see test_accelerator.test_calibration_fills_the_budget


def test_prepare_without_a_sample_uses_the_longest_input(make_local, monkeypatch):
    seen, real = [], FakeSentenceTransformer.encode_document
    monkeypatch.setattr(FakeSentenceTransformer, "encode_document", lambda s, t, **k: seen.extend(t) or real(s, t, **k))
    models = make_local("cuda")
    models.prepare(lambda: None)  # nothing left to embed meanwhile
    assert len(seen[0]) == 8 * models.embedder().max_seq_length  # longer than any input: it gets truncated


def test_prepare_off_cuda_asks_for_no_sample(fake_models):
    fake_models.prepare(lambda: pytest.fail("no calibration on the CPU, so no sample needed"))
    assert fake_models.batch == 128


@pytest.mark.parametrize("device, kind", [("cuda", OnGpu), ("mps", OnGpu), ("cpu", InRam)])
def test_vectors_live_where_the_models_compute(make_local, device, kind):
    vectors = make_local(device).vectors(np.arange(2), np.ones((2, 4), np.float16))
    assert isinstance(vectors, kind)
