from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
import torch
from huggingface_hub.utils import are_progress_bars_disabled
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Transformer
from transformers.utils import logging as transformers_logging

from sift.api.errors import ModelUnavailableError
from sift.inference.accelerator import CPU
from sift.inference.held import Embedder, Loaded, OnDevice, Ranker
from sift.inference.local import (
    BROKEN_REMEDY,
    UNLOADED_REMEDY,
    Local,
    loading,
    local,
    unloaded,
    width_of,
    within_memory,
)
from sift.inference.models import Models
from tests.models import Hashed
from tests.tiny_model import POSITIONS, WIDTH, tiny_models, tiny_prompted, tiny_routed, tiny_static


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_models(tmp_path_factory.mktemp("models"))


def test_a_model_from_a_folder_embeds_texts_as_normalised_halves_in_their_order(tiny: Path) -> None:
    models = local(str(tiny))
    assert models.name.startswith(f"{tiny.resolve()}#")
    texts = ["dark matter halo models", "pruning", "skips ranges", "halo"]
    rows = models.encode(texts, query=False)
    assert (rows.shape, rows.dtype) == ((4, WIDTH), np.float16)
    assert np.allclose(np.linalg.norm(rows.astype(np.float32), axis=1), 1, atol=1e-2)
    alone = models.encode(texts[1:2], query=False)
    assert np.allclose(alone, rows[1:2], atol=1e-2)


def test_the_model_loads_once_when_first_needed(tiny: Path) -> None:
    models = Local(str(tiny))
    assert not isinstance(models.state, OnDevice)
    models.prepare()
    loaded = placed(models).embedder
    models.encode(["halo"], query=True)
    assert (loaded is not None, placed(models).embedder is loaded) == (True, True)


def placed(models: Local) -> OnDevice:
    assert isinstance(models.state, OnDevice)
    return models.state


def embedder_of(models: Local) -> Loaded:
    loaded = placed(models).embedder
    assert loaded is not None
    return loaded


def ranker_of(models: Local) -> Ranker:
    ranker = placed(models).ranker
    assert ranker is not None
    return ranker


def hashed(_: Path) -> Models:
    return Hashed()


def local_in(folder: Path) -> Models:
    return local(str(folder), str(folder.parent / "reranker"))


@pytest.mark.parametrize("make", [hashed, local_in])
def test_the_fake_keeps_the_contract_the_local_models_keep(tiny: Path, make: Callable[[Path], Models]) -> None:
    models = make(tiny)
    for query in (False, True):
        rows = models.encode(["dark matter", "halo models", "pruning"], query=query)
        assert rows.dtype == np.float16
        assert rows.shape[0] == 3
        assert np.allclose(np.linalg.norm(rows.astype(np.float32), axis=1), 1, atol=1e-2)
        assert models.encode([], query=query).shape == (0, rows.shape[1])
    scores = models.rerank("dark matter", ["dark matter halo", "pruning skips ranges", "halo"])
    assert (scores.shape, scores.dtype) == ((3,), np.float32)
    assert bool(np.isfinite(scores).all())
    assert models.rerank("dark matter", []).shape == (0,)
    assert models.prepare() == models.prepare_ranker()
    assert models.name


def test_a_model_that_says_not_how_wide_its_vectors_are_is_refused() -> None:
    assert width_of("m", 768) == 768
    with pytest.raises(ValueError, match="m says not how wide its vectors are"):
        width_of("m", None)


def test_a_model_without_a_transformers_tokenizer_is_batched_by_characters(tmp_path: Path) -> None:
    models = local(str(tiny_static(tmp_path)))
    rows = models.encode(["dark matter", "halo"], query=False)
    assert rows.shape == (2, WIDTH)


def test_queries_and_documents_are_embedded_with_their_prompts_and_at_most_1024_tokens(tmp_path: Path) -> None:
    folder = tiny_prompted(tmp_path, 2048)
    models = Local(str(folder))
    texts = ["dark matter", "skips ranges"]
    queries, documents = models.encode(texts, query=True), models.encode(texts, query=False)
    own = cast("Embedder", SentenceTransformer(str(folder)))
    query = own.encode_query(texts, batch_size=2, normalize_embeddings=True, show_progress_bar=False)
    document = own.encode_document(texts, batch_size=2, normalize_embeddings=True, show_progress_bar=False)
    assert np.allclose(queries, np.asarray(query, np.float32), atol=1e-2)
    assert np.allclose(documents, np.asarray(document, np.float32), atol=1e-2)
    assert not np.allclose(queries, documents, atol=1e-2)
    assert embedder_of(models).model.max_seq_length == 1024


def test_a_model_that_cannot_be_loaded_says_what_to_check(tmp_path: Path) -> None:
    with pytest.raises(ModelUnavailableError, match="could not be loaded") as refused:
        local(str(tmp_path / "missing")).prepare()
    assert refused.value.remedy is not None
    assert "SIFT_EMBED_MODEL" in refused.value.remedy
    assert "\n" not in refused.value.message


def test_a_reranker_reading_fewer_than_1024_tokens_reads_passages_cut_to_its_length(tiny: Path) -> None:
    models = local(str(tiny), str(tiny.parent / "reranker"))
    scores = models.rerank("dark matter", [" ".join(["halo"] * 200), "pruning"])
    assert bool(np.isfinite(scores).all())
    assert ranker_of(models).max_seq_length == POSITIONS


def test_a_model_too_large_for_the_memory_left_says_so_not_to_check_the_network() -> None:
    full = unloaded("owner/model", torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"), "cuda")
    assert full.message == "the model owner/model could not be loaded: CUDA out of memory. Tried to allocate 2.00 GiB"
    assert full.remedy is not None
    assert "CUDA_VISIBLE_DEVICES" in full.remedy
    for device in ("cpu", "mps", "cuda"):
        remedy = unloaded("owner/model", MemoryError(), device).remedy
        assert remedy is not None
        assert "smaller model" in remedy
        assert "CUDA" not in remedy
    apple = unloaded("owner/model", RuntimeError("MPS backend out of memory (MPS allocated: 9 GiB)"), "mps").remedy
    assert apple is not None
    assert "CUDA" not in apple
    assert unloaded("owner/model", OSError("401 Client Error\nmore"), "cuda").remedy == UNLOADED_REMEDY
    assert unloaded("owner/model", ValueError("no kernel image is available"), "cuda").remedy == BROKEN_REMEDY


def _too_deep() -> int:
    raise RecursionError("maximum recursion depth exceeded")


def test_a_recursion_too_deep_while_a_model_loads_is_no_model_unavailable() -> None:
    with pytest.raises(RecursionError):
        loading("owner/model", CPU, _too_deep)
    assert loading("owner/model", CPU, lambda: 7) == 7


def test_a_model_out_of_memory_while_it_runs_says_so_with_what_frees_it() -> None:
    def full() -> int:
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")

    with pytest.raises(ModelUnavailableError, match="ran out of memory on the cuda") as refused:
        within_memory("owner/model", "cuda", full)
    assert refused.value.remedy is not None
    assert "CUDA_VISIBLE_DEVICES" in refused.value.remedy

    def broken() -> int:
        raise RuntimeError("expected all tensors to be on the same device")

    with pytest.raises(RuntimeError, match="same device"):
        within_memory("owner/model", "cuda", broken)
    assert within_memory("owner/model", "cpu", lambda: 7) == 7


def test_a_model_routing_queries_to_static_embeddings_caps_only_what_reads_tokens(tmp_path: Path) -> None:
    models = Local(str(tiny_routed(tmp_path)))
    assert models.encode(["dark matter"], query=True).shape == (1, WIDTH)
    assert models.encode(["dark matter"], query=False).shape == (1, WIDTH)
    assert [each.max_seq_length for each in embedder_of(models).model.modules() if isinstance(each, Transformer)] == [
        1024
    ]


class Predicting(Protocol):
    def predict(self, inputs: list[tuple[str, str]], *, activation_fn: torch.nn.Module | None = None) -> object: ...


def test_rerank_gives_raw_scores_whatever_the_model_applies(tiny: Path) -> None:
    folder = tiny.parent / "reranker"
    pairs = [("dark matter", "halo models"), ("dark matter", "pruning")]
    own = cast("Predicting", CrossEncoder(str(folder)))
    raw = np.asarray(own.predict(pairs, activation_fn=torch.nn.Identity()), np.float32)
    squashed = np.asarray(own.predict(pairs), np.float32)
    scores = local(str(tiny), str(folder)).rerank("dark matter", ["halo models", "pruning"])
    assert np.allclose(scores, raw, atol=1e-4)
    assert not np.allclose(scores, squashed, atol=1e-2)


def test_loading_the_models_draws_no_progress_bar_and_downloads_still_show_theirs(
    tiny: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models = local(str(tiny), str(tiny.parent / "reranker"))
    models.prepare()
    models.prepare_ranker()
    assert "Loading weights" not in capsys.readouterr().err
    assert not transformers_logging.is_progress_bar_enabled()
    assert not are_progress_bars_disabled()


def modules_of(model: object) -> list[torch.nn.Module]:
    return list(cast("torch.nn.Module", model).modules())


def test_the_models_run_in_inference_mode_as_models_loaded_for_it(tiny: Path) -> None:
    models = local(str(tiny), str(tiny.parent / "reranker"))
    models.prepare()
    models.prepare_ranker()
    seen: set[tuple[bool, bool]] = set()

    def heard(module: torch.nn.Module, _args: tuple[object, ...], _output: object) -> None:
        seen.add((torch.is_inference_mode_enabled(), module.training))

    for module in [*modules_of(embedder_of(models).model), *modules_of(placed(models).ranker)]:
        module.register_forward_hook(heard)
    models.encode(["halo"], query=False)
    models.rerank("dark matter", ["halo"])
    assert seen == {(True, False)}


def test_texts_are_embedded_longest_first_by_the_tokens_the_model_reads_not_their_characters(tiny: Path) -> None:
    models = Local(str(tiny), state=OnDevice(CPU, 0.0, batch=1))
    models.prepare()
    (reader,) = [each for each in embedder_of(models).model.modules() if isinstance(each, Transformer)]
    read: list[int] = []

    def heard(_module: torch.nn.Module, args: tuple[dict[str, torch.Tensor], ...]) -> None:
        read.append(int(args[0]["input_ids"].shape[1]))

    reader.register_forward_pre_hook(heard)
    models.encode(["pruningpruningpruningpruning", "halo halo halo halo halo"], query=False)
    assert read == [7, 3]
