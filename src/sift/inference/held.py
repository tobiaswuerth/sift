from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

import torch

from sift.inference.accelerator import Device

RERANK_BATCH = 16


class Tokenizer(Protocol):
    def __call__(self, texts: list[str], *, truncation: bool, max_length: int) -> Mapping[str, list[list[int]]]: ...


class Capped(Protocol):
    @property
    def max_seq_length(self) -> int | None: ...

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None: ...


class Movable(Protocol):
    def to(self, device: str) -> object: ...


class Embedder(Capped, Movable, Protocol):
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


class Ranker(Capped, Movable, Protocol):
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


@dataclass(frozen=True, slots=True)
class Unplaced:
    pass


@dataclass(frozen=True, slots=True)
class OnDevice:
    device: Device
    used: float
    embedder: Loaded | None = None
    ranker: Ranker | None = None
    batch: int | None = None
    rerank_batch: int = RERANK_BATCH


@dataclass(frozen=True, slots=True)
class InRam:
    embedder: Loaded | None
    ranker: Ranker | None
    batch: int | None
    rerank_batch: int


type HeldState = Unplaced | OnDevice | InRam


def movables(state: OnDevice | InRam) -> tuple[Movable, ...]:
    embedder = None if state.embedder is None else state.embedder.model
    return tuple(model for model in (embedder, state.ranker) if model is not None)


def rested(state: OnDevice) -> InRam:
    return InRam(state.embedder, state.ranker, state.batch, state.rerank_batch)


def woken(state: InRam, device: Device, now: float) -> OnDevice:
    return OnDevice(device, now, state.embedder, state.ranker, state.batch, state.rerank_batch)


def idle_since(state: HeldState, now: float, idle: float) -> OnDevice | None:
    return state if isinstance(state, OnDevice) and now - state.used >= idle else None
