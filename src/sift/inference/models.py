from collections.abc import Sequence
from typing import Protocol

from sift.domain.codec.codes import Halves


class Models(Protocol):
    @property
    def name(self) -> str: ...

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves: ...
