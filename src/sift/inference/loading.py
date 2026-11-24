import logging
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast

from sift.domain.codec.codes import Floats, Halves
from sift.inference.hub import pinned
from sift.inference.models import Models

if TYPE_CHECKING:
    from sift.inference.accelerator import Device
    from sift.inference.ocr import Ocr

REDIRECTS = "torch.distributed.elastic.multiprocessing.redirects"


class LocalLoader(Protocol):
    def local(self, model: str, reranker: str, gpus: str | None) -> Models: ...


def _models_module(name: str) -> object:
    logging.getLogger(REDIRECTS).setLevel(logging.ERROR)
    return import_module(name)


def _as_local(module: object) -> LocalLoader:
    return cast("LocalLoader", module)


def _local_module() -> LocalLoader:
    return _as_local(_models_module("sift.inference.local"))


@dataclass(slots=True)
class Deferred:
    model: str
    reranker: str
    gpus: str | None
    built: Models | None = None

    @property
    def name(self) -> str:
        return pinned(self.model)

    def _built(self) -> Models:
        if self.built is None:
            self.built = _local_module().local(self.model, self.reranker, self.gpus)
        return self.built

    def prepare(self) -> tuple[str, ...]:
        return self._built().prepare()

    def prepare_ranker(self) -> tuple[str, ...]:
        return self._built().prepare_ranker()

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        return self._built().encode(texts, query=query)

    def rerank(self, query: str, texts: Sequence[str]) -> Floats:
        return self._built().rerank(query, texts)


def deferred(model: str, reranker: str, gpus: str | None) -> Deferred:
    return Deferred(model, reranker, gpus)


class OcrModule(Protocol):
    def ocr_device(self) -> "Device": ...

    def loaded_ocr(self, device: "Device") -> "Ocr": ...

    def pages_text(self, ocr: "Ocr", pictures: Sequence[bytes]) -> list[str]: ...


def _as_ocr(module: object) -> OcrModule:
    return cast("OcrModule", module)


def ocr_module() -> OcrModule:
    return _as_ocr(_models_module("sift.inference.ocr"))
