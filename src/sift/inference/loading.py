from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

from sift.domain.codec.codes import Halves
from sift.inference.hub import pinned
from sift.inference.models import Models


class Local(Protocol):
    def local(self, model: str) -> Models: ...


def _as_local(module: object) -> Local:
    return cast("Local", module)


def _local_module() -> Local:
    return _as_local(import_module("sift.inference.local"))


@dataclass(slots=True)
class Deferred:
    model: str
    built: Models | None = None

    @property
    def name(self) -> str:
        return pinned(self.model)

    def _built(self) -> Models:
        if self.built is None:
            self.built = _local_module().local(self.model)
        return self.built

    def prepare(self) -> None:
        self._built().prepare()

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        return self._built().encode(texts, query=query)


def deferred(model: str) -> Deferred:
    return Deferred(model)
