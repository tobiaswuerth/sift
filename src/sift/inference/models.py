from collections.abc import Sequence
from typing import Protocol

from sift.domain.codec.codes import Floats, Halves


class Models(Protocol):
    @property
    def name(self) -> str: ...

    def prepare(self) -> tuple[str, ...]: ...

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves: ...

    def rerank(self, query: str, texts: Sequence[str]) -> Floats: ...
