from collections.abc import Sequence
from typing import Protocol

from sift.domain.codec.codes import Floats, Halves
from sift.inference.prepared import ModelsPrepared


class Models(Protocol):
    @property
    def name(self) -> str: ...

    def prepare(self) -> ModelsPrepared: ...

    def prepare_ranker(self) -> ModelsPrepared: ...

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves: ...

    def rerank(self, query: str, texts: Sequence[str]) -> Floats: ...
