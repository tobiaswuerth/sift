import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as transformers_logging

from sift.domain.codec.codes import Floats, Halves
from sift.inference.accelerator import Device, chosen, threads_for, too_large
from sift.inference.batches import Shape, in_batches
from sift.inference.hub import hub, pinned

MOST_TOKENS = 1024


class Tokenizer(Protocol):
    def __call__(self, texts: list[str], *, truncation: bool, max_length: int) -> Mapping[str, list[list[int]]]: ...


class Embedder(Protocol):
    @property
    def tokenizer(self) -> Tokenizer: ...

    @property
    def max_seq_length(self) -> int | None: ...

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None: ...

    def get_embedding_dimension(self) -> int | None: ...

    def encode_query(
        self, inputs: list[str], *, batch_size: int, normalize_embeddings: bool, show_progress_bar: bool
    ) -> object: ...

    def encode_document(
        self, inputs: list[str], *, batch_size: int, normalize_embeddings: bool, show_progress_bar: bool
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class Loaded:
    model: Embedder
    device: Device
    width: int


def width_of(name: str, found: int | None) -> int:
    if found is None:
        raise ValueError(f"{name} says not how wide its vectors are: name an embedding model")
    return found


def _loaded(name: str) -> Loaded:
    transformers_logging.disable_progress_bar()
    device = chosen(torch.cuda, mps=torch.backends.mps.is_available())
    torch.set_num_threads(threads_for(device, os.cpu_count() or 1))
    repository, revision = hub(name)
    model = cast(
        "Embedder",
        SentenceTransformer(repository, device=device.name, revision=revision, model_kwargs={"dtype": device.dtype}),
    )
    model.max_seq_length = min(model.max_seq_length or MOST_TOKENS, MOST_TOKENS)
    return Loaded(model, device, width_of(name, model.get_embedding_dimension()))


def _tokens(model: Embedder, texts: list[str]) -> list[list[int]]:
    return model.tokenizer(texts, truncation=True, max_length=MOST_TOKENS)["input_ids"] if texts else []


def _run(model: Embedder, texts: list[str], *, query: bool) -> Floats:
    encode = model.encode_query if query else model.encode_document
    with torch.inference_mode():
        found = encode(texts, batch_size=len(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(found, np.float32)


@dataclass(slots=True)
class Local:
    model: str
    loaded: Loaded | None = None

    @property
    def name(self) -> str:
        return pinned(self.model)

    def _ready(self) -> Loaded:
        if self.loaded is None:
            self.loaded = _loaded(self.model)
        return self.loaded

    def prepare(self) -> None:
        self._ready()

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        loaded = self._ready()

        def run(part: list[str]) -> Floats:
            return _run(loaded.model, part, query=query)

        lengths = [len(ids) for ids in _tokens(loaded.model, list(texts))]
        return in_batches(texts, lengths, Shape(loaded.width, loaded.device.batch), run, too_large)


def local(model: str) -> Local:
    return Local(model)
