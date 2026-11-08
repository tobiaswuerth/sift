"""The models of this machine: an embedding model and a reranker, loaded once per precision, on the GPU while in use.
Sentence Transformers, GPU memory and the clock are fakes."""

import threading
import time

import numpy as np
import pytest

from sift import accelerator, local

from .conftest import FakeCrossEncoder, FakeSentenceTransformer, GiB


def test_models_load_once_with_limits(fake_models):
    model = fake_models.embedder()
    assert (model.name, model.device, model.max_seq_length, model.dtype) == (
        "google/embeddinggemma-2",
        "cpu",
        local.MAX_TOKENS,
        "float32",
    )
    assert fake_models.reranker().name == "Alibaba-NLP/gte-reranker-modernbert-base"
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
    monkeypatch.setattr(FakeSentenceTransformer, "max_batch", 20)  # rows the fake GPU has memory for
    models = make_local("cuda")
    assert models.encode([f"text {i}" for i in range(40)]).shape == (40, 64)
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


def test_prepare_calibrates_with_the_longest_input_the_model_takes(make_local, gpu_memory):
    # A batch that fits there fits any text, whatever the model: none needs more memory than the longest input. The
    # longest text still to embed, by characters, was read from every pending chunk at the start, and could have fewer
    # tokens than others: it ran out of memory at once. Larger batches of short texts were slower (more padding).
    models = make_local("cuda")
    models.prepare()
    model = models.embedder()
    assert [len(batch) for batch in model.embedded] == gpu_memory.batches
    assert {text for batch in model.embedded for text in batch} == {"x " * (4 * model.max_seq_length)}
    assert len(model.embedded[0][0]) == 8 * model.max_seq_length  # longer than any input: it gets truncated
    assert models.batch == 294  # see test_accelerator.test_calibration_fills_the_budget


def test_documents_and_queries_get_the_models_prompts_and_tasks(fake_models, monkeypatch):
    # As Sentence Transformers' encode_document and encode_query pick them, from the model's own configuration.
    monkeypatch.setattr(FakeSentenceTransformer, "prompts", {"query": "q: ", "document": "d: ", "passage": "p: "})
    fake_models.encode(["brush"])
    fake_models.encode(["brush"], query=True)
    monkeypatch.setattr(FakeSentenceTransformer, "prompts", {"passage": "p: "})
    fake_models.encode(["brush"])
    fake_models.encode(["brush"], query=True)
    assert fake_models.embedder().texts == [
        ("d: brush", "document"),
        ("q: brush", "query"),
        ("p: brush", "document"),
        ("brush", "query"),  # no prompt for queries in this model
    ]


def test_the_model_embeds_as_in_inference_not_training(fake_models):
    # In training mode, a model's dropout would change every vector at random.
    assert not fake_models.embedder().training


def test_the_next_batch_is_tokenized_while_the_gpu_embeds_one(make_local):
    # The GPU waited while each batch was tokenized: busy 80% of the time.
    models = make_local("cuda")
    model = models.embedder()
    batches = iter([["brush strokes"], ["pigments on canvas"], ["reward shaping"]])
    vectors = models.encode_batches(batches)
    first = next(vectors)
    assert [text for text, *_ in model.texts] == ["brush strokes", "pigments on canvas"]  # the second tokenized
    assert model.embedded == [["brush strokes"]]  # before the vectors of the first were collected
    assert first.tolist() == models.encode(["brush strokes"]).tolist()
    assert [v.tolist() for v in vectors] == [
        models.encode([t]).tolist() for t in ("pigments on canvas", "reward shaping")
    ]


def test_a_batch_out_of_memory_in_a_stream_is_embedded_in_smaller_parts(make_local, monkeypatch):
    monkeypatch.setattr(FakeSentenceTransformer, "max_batch", 3)
    models = make_local("cuda")
    batches = [["a", "b"], ["c", "d", "e", "f"], ["g", "h", "i", "j"], ["k"]]
    got = list(models.encode_batches(iter(batches)))
    assert [g.tolist() for g in got] == [models.encode(b).tolist() for b in batches]
    assert models.batch == 2  # halved, as it ran out of memory: the batches after it in parts of the smaller size


def test_lengths_are_tokens_as_the_model_reads_them(fake_models, monkeypatch):
    monkeypatch.setattr(local, "LENGTHS_AT_ONCE", 2)  # several rounds
    texts = ["brush", "brush strokes on canvas", "word " * 5000, ""]
    assert fake_models.lengths(texts) == [3, 6, local.MAX_TOKENS, 2]  # a token per word and two special ones, cut
    assert fake_models.lengths([]) == []


def test_lengths_are_counted_without_pytorch(fake_models, monkeypatch):
    # They're counted in another thread than the models' (see embed): PyTorch crashes in other threads on Python 3.15,
    # and the model's own tokenizer, which Sentence Transformers sets to pad its batches, must not be shared.
    fake_models.prepare()
    monkeypatch.setattr(FakeSentenceTransformer, "preprocess", lambda *a, **k: pytest.fail("no PyTorch tensors"))
    found = []
    thread = threading.Thread(target=lambda: found.append(fake_models.lengths(["brush strokes", "pigments"])))
    thread.start()
    thread.join(timeout=10)
    assert found == [[4, 3]]


def test_lengths_of_a_model_without_a_fast_tokenizer_are_characters(fake_models, monkeypatch):
    # A static embedding model's tokens come without one. The lengths only group texts into batches, so characters do.
    monkeypatch.setattr(FakeSentenceTransformer, "tokenizer", None)
    assert fake_models.lengths(["brush", "brush strokes"]) == [5, 13]


def test_the_models_load_once_when_threads_ask_at_once(make_local, monkeypatch):
    loads, real = [], local._embedder

    def slow(name, dtype):
        loads.append(name)
        time.sleep(0.2)
        return real(name, dtype)

    monkeypatch.setattr(local, "_embedder", slow)
    models = make_local()
    threads = [threading.Thread(target=models.embedder) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert len(loads) == 1


def test_prepare_off_cuda_calibrates_nothing(fake_models):
    fake_models.prepare()
    assert fake_models.batch == 128
    assert fake_models.embedder().embedded == []
