from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Protocol, cast

import numpy as np
import torch
from huggingface_hub.utils import enable_progress_bars
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Transformer
from transformers import PreTrainedTokenizerBase
from transformers.utils import logging as transformers_logging

from sift.api.errors import ModelUnavailableError
from sift.domain.codec.codes import Floats, Halves
from sift.inference.accelerator import Device, Found, chosen, emptied, host_full, nvidia, too_large
from sift.inference.batches import Attempt, Shape, in_batches
from sift.inference.hub import RERANK_MODEL, hub, pinned

MOST_TOKENS = 1024
RERANK_BATCH = 16
BROKEN_REMEDY = "check the name in SIFT_EMBED_MODEL or SIFT_RERANK_MODEL; if it is right, report this with the name"
UNLOADED_REMEDY = (
    "check the network and the name in SIFT_EMBED_MODEL or SIFT_RERANK_MODEL; for a gated model, log in: hf auth login"
)
SMALLER_REMEDY = "free memory others hold, or name a smaller model in SIFT_EMBED_MODEL or SIFT_RERANK_MODEL"
FULL_REMEDIES = MappingProxyType(
    {"cuda": "free the GPU's memory others hold, or set CUDA_VISIBLE_DEVICES=-1 to run the models on the CPU"}
)


class Tokenizer(Protocol):
    def __call__(self, texts: list[str], *, truncation: bool, max_length: int) -> Mapping[str, list[list[int]]]: ...


class Capped(Protocol):
    @property
    def max_seq_length(self) -> int | None: ...

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None: ...


class Embedder(Capped, Protocol):
    @property
    def tokenizer(self) -> object: ...

    def modules(self) -> Iterator[object]: ...

    def get_embedding_dimension(self) -> int | None: ...

    def encode_query(
        self, inputs: list[str], *, batch_size: int, normalize_embeddings: bool, show_progress_bar: bool
    ) -> object: ...

    def encode_document(
        self, inputs: list[str], *, batch_size: int, normalize_embeddings: bool, show_progress_bar: bool
    ) -> object: ...


class Ranker(Capped, Protocol):
    @property
    def tokenizer(self) -> object: ...

    def predict(
        self,
        inputs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        activation_fn: torch.nn.Module,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class Loaded:
    model: Embedder
    width: int


def width_of(name: str, found: int | None) -> int:
    if found is None:
        raise ValueError(f"{name} says not how wide its vectors are: name an embedding model")
    return found


def _device() -> Device:
    transformers_logging.disable_progress_bar()
    enable_progress_bars()
    found = Found(torch.backends.mps.is_available(), nvidia(), torch.version.cuda)
    device = chosen(torch.cuda, found)
    torch.set_num_threads(device.threads)
    return device


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
    except (RuntimeError, MemoryError, OSError) as error:
        if not too_large(error):
            raise
        said = f"the model {name} ran out of memory on the {device}: {_first_line(error)}"
        raise ModelUnavailableError(said, _full(error, device)) from error


def _loading[T](name: str, device: Device, load: Callable[[], T]) -> T:
    try:
        return load()
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

    return _loading(name, device, load)


def _ranker(name: str, device: Device) -> Ranker:
    repository, revision = hub(name)

    def load() -> Ranker:
        model = CrossEncoder(repository, device=device.name, revision=revision, model_kwargs={"dtype": device.dtype})
        return _capped(cast("Ranker", model))

    return _loading(name, device, load)


def _counting(tokenizer: object) -> Tokenizer:
    return cast("Tokenizer", tokenizer)


def _lengths(tokenizer: object, texts: list[str]) -> list[int]:
    if not isinstance(tokenizer, PreTrainedTokenizerBase) or not texts:
        return [len(text) for text in texts]
    counted = _counting(tokenizer)(texts, truncation=True, max_length=MOST_TOKENS)
    return [len(ids) for ids in counted["input_ids"]]


def _run(model: Embedder, texts: list[str], *, query: bool) -> Floats:
    encode = model.encode_query if query else model.encode_document
    with torch.inference_mode():
        found = encode(texts, batch_size=len(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(found, np.float32)


def _scored(ranker: Ranker, query: str, texts: list[str]) -> Floats:
    pairs = [(query, text) for text in texts]
    with torch.inference_mode():
        found = ranker.predict(pairs, batch_size=len(pairs), show_progress_bar=False, activation_fn=torch.nn.Identity())
    return np.asarray(found, np.float32).reshape(-1, 1)


@dataclass(slots=True)
class Local:
    model: str
    reranker: str = RERANK_MODEL
    device: Device | None = None
    embedder: Loaded | None = None
    ranker: Ranker | None = None
    batch: int | None = None
    rerank_batch: int = RERANK_BATCH

    @property
    def name(self) -> str:
        return pinned(self.model)

    def _on(self) -> Device:
        if self.device is None:
            self.device = _device()
        return self.device

    def _embedding(self) -> Loaded:
        if self.embedder is None:
            self.embedder = _embedder(self.model, self._on())
        return self.embedder

    def _ranking(self) -> Ranker:
        if self.ranker is None:
            self.ranker = _ranker(self.reranker, self._on())
        return self.ranker

    def prepare(self) -> tuple[str, ...]:
        self._embedding()
        notice = self._on().notice
        return () if notice is None else (notice,)

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        loaded, device = self._embedding(), self._on()

        def run(part: list[str]) -> Floats:
            return _run(loaded.model, part, query=query)

        shape = Shape(loaded.width, self.batch or device.batch)
        lengths = _lengths(loaded.model.tokenizer, list(texts))
        found = within_memory(self.model, device.name, partial(in_batches, texts, lengths, shape, self._attempt(run)))
        self.batch = found.batch
        return found.rows.astype(np.float16)

    def rerank(self, query: str, texts: Sequence[str]) -> Floats:
        ranker = self._ranking()

        def run(part: list[str]) -> Floats:
            return _scored(ranker, query, part)

        shape = Shape(1, self.rerank_batch)
        lengths = _lengths(ranker.tokenizer, list(texts))
        attempts = partial(in_batches, texts, lengths, shape, self._attempt(run))
        found = within_memory(self.reranker, self._on().name, attempts)
        self.rerank_batch = found.batch
        return found.rows.reshape(-1)

    def _attempt(self, run: Callable[[list[str]], Floats]) -> Attempt:
        device = self._on()
        return Attempt(run, too_large, lambda: emptied(device))


def local(model: str, reranker: str = RERANK_MODEL) -> Local:
    return Local(model, reranker)
