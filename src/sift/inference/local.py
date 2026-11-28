import importlib.util
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from types import MappingProxyType
from typing import NamedTuple, Protocol, cast

import numpy as np
import torch
from huggingface_hub.utils import enable_progress_bars
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Transformer
from transformers import PreTrainedTokenizerBase
from transformers.utils import logging as transformers_logging

from sift.api.errors import ModelUnavailableError
from sift.domain.codec.codes import MOST_BITS, Floats, Halves
from sift.inference.accelerator import CACHES, MACHINE, Device, Machine, emptied, host_full, placed, too_large
from sift.inference.batches import Attempt, BatchShape, in_batches
from sift.inference.held import (
    Capped,
    Embedder,
    Forward,
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
from sift.inference.pairing import read_unpaired
from sift.inference.prepared import ModelsPrepared

MOST_TOKENS = 1024
QUERY_PROMPTS = ("query",)
DOCUMENT_PROMPTS = ("document", "passage", "corpus")
IDLE_SECONDS = 300.0
RESTS = "cuda"
PINNED = frozenset({"cuda"})
WARM_TEXT = "word " * 256
COMPILED = "{model} was compiled for the GPU in {seconds:.0f} s, as each process does before it embeds many passages"
UNCOMPILED = "{model} could not be compiled, so it runs as it is: {problem}"
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
    if found > MOST_BITS:
        raise ValueError(f"{name} makes vectors of {found:,} numbers, and sift codes at most {MOST_BITS:,}")
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


class Loads[T](Protocol):
    def __call__(self, *, local_files_only: bool) -> T: ...


def cache_first[T](load: Loads[T]) -> T:
    try:
        return load(local_files_only=True)
    except OSError:
        return load(local_files_only=False)


def _embedder(name: str, device: Device) -> Loaded:
    repository, revision = hub(name)

    def load(*, local_files_only: bool) -> Loaded:
        model = SentenceTransformer(
            repository,
            device=device.name,
            revision=revision,
            model_kwargs={"dtype": device.dtype},
            local_files_only=local_files_only,
        )
        held = _reading_capped(cast("Embedder", cast("object", model.eval())))
        return Loaded(held, width_of(name, held.get_embedding_dimension()))

    return loading(name, device, partial(cache_first, load))


def _ranker(name: str, device: Device) -> Ranker:
    pinned(name)
    repository, revision = hub(name)

    def load(*, local_files_only: bool) -> Ranker:
        model = CrossEncoder(
            repository,
            device=device.name,
            revision=revision,
            model_kwargs={"dtype": device.dtype},
            local_files_only=local_files_only,
        )
        return _capped(cast("Ranker", model))

    return loading(name, device, partial(cache_first, load))


def _counting(tokenizer: object) -> Tokenizer:
    return cast("Tokenizer", tokenizer)


def _lengths(tokenizer: object, texts: list[str]) -> list[int]:
    if not isinstance(tokenizer, PreTrainedTokenizerBase) or not texts:
        return [len(text) for text in texts]
    counted = _counting(tokenizer)(texts, truncation=True, max_length=MOST_TOKENS)
    return [len(ids) for ids in counted["input_ids"]]


def _prompt(model: Embedder, *, query: bool) -> str | None:
    named = next((name for name in (QUERY_PROMPTS if query else DOCUMENT_PROMPTS) if name in model.prompts), None)
    name = named or model.default_prompt_name
    return None if name is None else model.prompts.get(name)


def _task(*, query: bool) -> str:
    return "query" if query else "document"


class Tokenized(NamedTuple):
    features: dict[str, object]
    size: int


def _pinned(value: object) -> object:
    return value.pin_memory() if isinstance(value, torch.Tensor) else value


def _prepared(model: Embedder, texts: list[str], *, query: bool, pinned: bool) -> Tokenized:
    found = model.preprocess(texts, prompt=_prompt(model, query=query), task=_task(query=query))
    return Tokenized({key: _pinned(value) for key, value in found.items()} if pinned else found, len(texts))


def _placed(value: object, device: torch.device) -> object:
    return value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value


def _run(model: Embedder, forward: Forward, prepared: Tokenized, *, query: bool) -> Floats:
    features = {key: _placed(value, model.device) for key, value in prepared.features.items()}
    with torch.inference_mode():
        found = forward(features, task=_task(query=query))["sentence_embedding"]
        cut = found if model.truncate_dim is None else found[:, : model.truncate_dim]
        return torch.nn.functional.normalize(cut, p=2, dim=1).float().cpu().numpy()


def _compiled(model: Embedder) -> Forward:
    return cast("Forward", torch.compile(model, dynamic=True))


def _compiles(device: Device) -> bool:
    return device.name == "cuda" and importlib.util.find_spec("triton") is not None


def _scored(ranker: Ranker, query: str, texts: list[str], batch: int) -> Floats:
    pairs = [(query, text) for text in texts]
    found = ranker.predict(pairs, batch_size=batch, show_progress_bar=False, activation_fn=torch.nn.Identity())
    return np.asarray(found, np.float32).reshape(-1, 1)


@dataclass(frozen=True, slots=True)
class LocalHost:
    machine: Machine = MACHINE
    clock: Callable[[], float] = time.perf_counter
    embedders: Callable[[str, Device], Loaded] = _embedder
    rankers: Callable[[str, Device], Ranker] = _ranker
    caches: Mapping[str, Callable[[], None]] = CACHES
    compiler: Callable[[Embedder], Forward] = _compiled
    compiles: Callable[[Device], bool] = _compiles
    unpaired: Callable[[], tuple[str, ...]] = read_unpaired


LOCAL_HOST = LocalHost()


def _moved(state: OnDevice | InRam, device: str) -> None:
    for model in movables(state):
        model.to(device)


def _attempt[P](
    device: Device,
    caches: Mapping[str, Callable[[], None]],
    prepare: Callable[[list[str]], P],
    run: Callable[[P], Floats],
) -> Attempt[P]:
    return Attempt(prepare, run, too_large, partial(emptied, device, caches))


def _texts(texts: list[str]) -> list[str]:
    return texts


def _woken(name: str, host: LocalHost, state: InRam, where: tuple[Device, float]) -> OnDevice:
    device, now = where
    if device.name != RESTS:
        return OnDevice(device, now)
    try:
        within_memory(name, device.name, partial(_moved, state, device.name))
    except BaseException:
        _moved(state, RAM)
        emptied(device, host.caches)
        raise
    return woken(state, device, now)


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
            placed = _woken(self.model, self.host, state, (this_device(self.gpus, self.host.machine), now))
        else:
            placed = OnDevice(this_device(self.gpus, self.host.machine), now)
        self.state = placed
        return placed

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

    def prepare_many(self) -> ModelsPrepared:
        state, loaded = self._embedding(self._on())
        if loaded.compiled is not None or not self.host.compiles(state.device):
            return ModelsPrepared((), built=False)
        started = self.host.clock()
        rows = state.batch or state.device.batch
        self.host.machine.threads(state.device.threads)
        try:
            compiled = self.host.compiler(loaded.model)
            warm = _prepared(loaded.model, [WARM_TEXT] * rows, query=False, pinned=state.device.name in PINNED)
            _run(loaded.model, compiled, warm, query=False)
        except RecursionError:
            raise
        except RuntimeError as error:
            said = UNCOMPILED.format(model=self.name, problem=_first_line(error))
            return ModelsPrepared((said, *self.host.unpaired()), built=False)
        self.state = replace(state, embedder=replace(loaded, compiled=compiled))
        seconds = self.host.clock() - started
        return ModelsPrepared((COMPILED.format(model=self.name, seconds=seconds),), built=True)

    def prepare_ranker(self) -> ModelsPrepared:
        held = isinstance(self.state, OnDevice) and self.state.ranker is not None
        state, _ = self._ranking(self._on())
        return ModelsPrepared(_notices(state.device), built=not held)

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        state, loaded = self._embedding(self._on())

        compiled = None if query else loaded.compiled

        def run(prepared: Tokenized) -> Floats:
            forward = compiled if compiled is not None and prepared.size > 1 else loaded.model
            return _run(loaded.model, forward, prepared, query=query)

        prepare = partial(_prepared, loaded.model, query=query, pinned=state.device.name in PINNED)
        shape = BatchShape(loaded.width, state.batch or state.device.batch)
        lengths = _lengths(loaded.model.tokenizer, list(texts))
        attempt = _attempt(state.device, self.host.caches, prepare, run)
        attempts = partial(in_batches, texts, lengths, shape, attempt)
        self.host.machine.threads(state.device.threads)
        found = within_memory(self.model, state.device.name, attempts)
        self.state = replace(state, batch=found.batch)
        return found.rows.astype(np.float16)

    def rerank(self, query: str, texts: Sequence[str]) -> Floats:
        state, ranker = self._ranking(self._on())

        def run(part: list[str]) -> Floats:
            return _scored(ranker, query, part, min(len(part), state.rerank_batch))

        shape = BatchShape(1, max(1, len(texts)))
        lengths = [len(text) for text in texts]
        attempts = partial(in_batches, texts, lengths, shape, _attempt(state.device, self.host.caches, _texts, run))
        self.host.machine.threads(state.device.threads)
        found = within_memory(self.reranker, state.device.name, attempts)
        halved = found.batch < shape.batch
        self.state = replace(state, rerank_batch=min(state.rerank_batch, found.batch) if halved else state.rerank_batch)
        return found.rows.reshape(-1)


def local(model: str, reranker: str = RERANK_MODEL, gpus: str | None = None) -> Local:
    return Local(model, reranker, gpus)
