import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from types import MappingProxyType
from typing import cast

import numpy as np
import torch
from huggingface_hub.utils import enable_progress_bars
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Transformer
from transformers import PreTrainedTokenizerBase
from transformers.utils import logging as transformers_logging

from sift.api.errors import ModelUnavailableError
from sift.domain.codec.codes import Floats, Halves
from sift.inference.accelerator import CACHES, MACHINE, Device, Machine, emptied, host_full, placed, too_large
from sift.inference.batches import Attempt, BatchShape, in_batches
from sift.inference.held import (
    Capped,
    Embedder,
    HeldState,
    InRam,
    Loaded,
    OnDevice,
    Ranker,
    Tokenizer,
    Unplaced,
    idle_since,
    movables,
    rested,
    woken,
)
from sift.inference.hub import RERANK_MODEL, hub, pinned
from sift.inference.prepared import ModelsPrepared

MOST_TOKENS = 1024
IDLE_SECONDS = 300.0
RESTS = "cuda"
RAM = "cpu"
BROKEN_REMEDY = "check the name in SIFT_EMBED_MODEL or SIFT_RERANK_MODEL; if it is right, report this with the name"
UNLOADED_REMEDY = (
    "check the network and the name in SIFT_EMBED_MODEL or SIFT_RERANK_MODEL; for a gated model, log in: hf auth login"
)
SMALLER_REMEDY = "free memory others hold, or name a smaller model in SIFT_EMBED_MODEL or SIFT_RERANK_MODEL"
FULL_REMEDIES = MappingProxyType(
    {"cuda": "free the GPU's memory others hold, or set CUDA_VISIBLE_DEVICES=-1 to run the models on the CPU"}
)


def width_of(name: str, found: int | None) -> int:
    if found is None:
        raise ValueError(f"{name} says not how wide its vectors are: name an embedding model")
    return found


def this_device(gpus: str | None = None, machine: Machine = MACHINE) -> Device:
    transformers_logging.disable_progress_bar()
    enable_progress_bars()
    return placed(machine, gpus)


def _capped[T: Capped](model: T) -> T:
    model.max_seq_length = min(model.max_seq_length or MOST_TOKENS, MOST_TOKENS)
    return model


def _reading_capped(model: Embedder) -> Embedder:
    for module in model.modules():
        if isinstance(module, Transformer):
            _capped(module)
    return model


def _first_line(error: BaseException) -> str:
    return str(error).strip().split("\n")[0] or type(error).__name__


def _full(error: BaseException, device: str) -> str:
    return SMALLER_REMEDY if host_full(error) else FULL_REMEDIES.get(device, SMALLER_REMEDY)


def unloaded(name: str, error: BaseException, device: str) -> ModelUnavailableError:
    if too_large(error):
        remedy = _full(error, device)
    else:
        remedy = UNLOADED_REMEDY if isinstance(error, OSError) else BROKEN_REMEDY
    return ModelUnavailableError(f"the model {name} could not be loaded: {_first_line(error)}", remedy)


def within_memory[T](name: str, device: str, run: Callable[[], T]) -> T:
    try:
        return run()
    except RecursionError:
        raise
    except (RuntimeError, MemoryError, OSError) as error:
        if not too_large(error):
            raise
        said = f"the model {name} ran out of memory on the {device}: {_first_line(error)}"
        raise ModelUnavailableError(said, _full(error, device)) from error


def loading[T](name: str, device: Device, load: Callable[[], T]) -> T:
    try:
        return load()
    except RecursionError:
        raise
    except (OSError, ValueError, RuntimeError, MemoryError) as error:
        raise unloaded(name, error, device.name) from error


def _embedder(name: str, device: Device) -> Loaded:
    repository, revision = hub(name)

    def load() -> Loaded:
        model = SentenceTransformer(
            repository, device=device.name, revision=revision, model_kwargs={"dtype": device.dtype}
        )
        held = _reading_capped(cast("Embedder", model))
        return Loaded(held, width_of(name, held.get_embedding_dimension()))

    return loading(name, device, load)


def _ranker(name: str, device: Device) -> Ranker:
    repository, revision = hub(name)

    def load() -> Ranker:
        model = CrossEncoder(repository, device=device.name, revision=revision, model_kwargs={"dtype": device.dtype})
        return _capped(cast("Ranker", model))

    return loading(name, device, load)


def _counting(tokenizer: object) -> Tokenizer:
    return cast("Tokenizer", tokenizer)


def _lengths(tokenizer: object, texts: list[str]) -> list[int]:
    if not isinstance(tokenizer, PreTrainedTokenizerBase) or not texts:
        return [len(text) for text in texts]
    counted = _counting(tokenizer)(texts, truncation=True, max_length=MOST_TOKENS)
    return [len(ids) for ids in counted["input_ids"]]


def _run(model: Embedder, texts: list[str], *, query: bool) -> Floats:
    encode = model.encode_query if query else model.encode_document
    found = encode(texts, batch_size=len(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(found, np.float32)


def _scored(ranker: Ranker, query: str, texts: list[str]) -> Floats:
    pairs = [(query, text) for text in texts]
    found = ranker.predict(pairs, batch_size=len(pairs), show_progress_bar=False, activation_fn=torch.nn.Identity())
    return np.asarray(found, np.float32).reshape(-1, 1)


@dataclass(frozen=True, slots=True)
class LocalHost:
    machine: Machine = MACHINE
    clock: Callable[[], float] = time.perf_counter
    embedders: Callable[[str, Device], Loaded] = _embedder
    rankers: Callable[[str, Device], Ranker] = _ranker
    caches: Mapping[str, Callable[[], None]] = CACHES


LOCAL_HOST = LocalHost()


def _moved(state: OnDevice | InRam, device: str) -> None:
    for model in movables(state):
        model.to(device)


def _attempt(device: Device, caches: Mapping[str, Callable[[], None]], run: Callable[[list[str]], Floats]) -> Attempt:
    return Attempt(run, too_large, partial(emptied, device, caches))


def _notices(device: Device) -> tuple[str, ...]:
    return () if device.notice is None else (device.notice,)


@dataclass(slots=True)
class Local:
    model: str
    reranker: str = RERANK_MODEL
    gpus: str | None = None
    host: LocalHost = LOCAL_HOST
    state: HeldState = field(default_factory=Unplaced)

    @property
    def name(self) -> str:
        return pinned(self.model)

    def _on(self) -> OnDevice:
        now, state = self.host.clock(), self.state
        if isinstance(state, OnDevice):
            placed = replace(state, used=now)
        elif isinstance(state, InRam):
            placed = self._woken(state, this_device(self.gpus, self.host.machine), now)
        else:
            placed = OnDevice(this_device(self.gpus, self.host.machine), now)
        self.state = placed
        return placed

    def _woken(self, state: InRam, device: Device, now: float) -> OnDevice:
        if device.name != RESTS:
            return OnDevice(device, now)
        try:
            within_memory(self.model, device.name, partial(_moved, state, device.name))
        except BaseException:
            _moved(state, RAM)
            emptied(device, self.host.caches)
            raise
        return woken(state, device, now)

    def release_idle(self) -> bool:
        state = idle_since(self.state, self.host.clock(), IDLE_SECONDS)
        if state is not None and state.device.busy:
            self.state = Unplaced()
            return True
        if state is None or state.device.name != RESTS:
            return False
        try:
            _moved(state, RAM)
        except BaseException:
            _moved(state, state.device.name)
            raise
        emptied(state.device, self.host.caches)
        self.state = rested(state)
        return True

    def _embedding(self, state: OnDevice) -> tuple[OnDevice, Loaded]:
        loaded = state.embedder or self.host.embedders(self.model, state.device)
        placed = replace(state, embedder=loaded)
        self.state = placed
        return placed, loaded

    def _ranking(self, state: OnDevice) -> tuple[OnDevice, Ranker]:
        ranker = state.ranker or self.host.rankers(self.reranker, state.device)
        placed = replace(state, ranker=ranker)
        self.state = placed
        return placed, ranker

    def prepare(self) -> ModelsPrepared:
        held = isinstance(self.state, OnDevice) and self.state.embedder is not None
        state, _ = self._embedding(self._on())
        return ModelsPrepared(_notices(state.device), built=not held)

    def prepare_ranker(self) -> ModelsPrepared:
        held = isinstance(self.state, OnDevice) and self.state.ranker is not None
        state, _ = self._ranking(self._on())
        return ModelsPrepared(_notices(state.device), built=not held)

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        state, loaded = self._embedding(self._on())

        def run(part: list[str]) -> Floats:
            return _run(loaded.model, part, query=query)

        shape = BatchShape(loaded.width, state.batch or state.device.batch)
        lengths = _lengths(loaded.model.tokenizer, list(texts))
        attempts = partial(in_batches, texts, lengths, shape, _attempt(state.device, self.host.caches, run))
        self.host.machine.threads(state.device.threads)
        found = within_memory(self.model, state.device.name, attempts)
        self.state = replace(state, batch=found.batch)
        return found.rows.astype(np.float16)

    def rerank(self, query: str, texts: Sequence[str]) -> Floats:
        state, ranker = self._ranking(self._on())

        def run(part: list[str]) -> Floats:
            return _scored(ranker, query, part)

        shape = BatchShape(1, state.rerank_batch)
        lengths = _lengths(ranker.tokenizer, list(texts))
        attempts = partial(in_batches, texts, lengths, shape, _attempt(state.device, self.host.caches, run))
        self.host.machine.threads(state.device.threads)
        found = within_memory(self.reranker, state.device.name, attempts)
        self.state = replace(state, rerank_batch=found.batch)
        return found.rows.reshape(-1)


def local(model: str, reranker: str = RERANK_MODEL, gpus: str | None = None) -> Local:
    return Local(model, reranker, gpus)
