import logging
import threading
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from sift.domain.codec.codes import Floats, Halves
from sift.inference.hub import pinned
from sift.inference.models import Models
from sift.inference.prepared import ModelsPrepared
from sift.late import late

if TYPE_CHECKING:
    from sift.inference.accelerator import Device
    from sift.inference.ocr import Ocr

REDIRECTS = "torch.distributed.elastic.multiprocessing.redirects"
LOCAL = "sift.inference.local"
OCR = "sift.inference.ocr"
VIRTUAL_MAC = "Skipping device"


class Resting(Models, Protocol):
    def release_idle(self) -> bool: ...


class HeldModels(Resting, Protocol):
    def close(self) -> None: ...


class LocalLoader(Protocol):
    def local(self, model: str, reranker: str, gpus: str | None) -> Resting: ...


def _models_module[P](name: str, protocol: type[P]) -> P:
    logging.getLogger(REDIRECTS).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", VIRTUAL_MAC, UserWarning)
    return late(name, protocol)


def _local_module() -> LocalLoader:
    return _models_module(LOCAL, LocalLoader)


@dataclass(slots=True, eq=False)
class Deferred:
    model: str
    reranker: str
    gpus: str | None
    built: Resting | None = None
    loader: Callable[[], LocalLoader] = _local_module
    lock: threading.Lock = field(default_factory=threading.Lock)
    named: str | None = None

    @property
    def name(self) -> str:
        if self.named is None:
            self.named = pinned(self.model)
        return self.named

    def _built(self) -> Resting:
        if self.built is None:
            self.built = self.loader().local(self.model, self.reranker, self.gpus)
        return self.built

    def prepare(self) -> ModelsPrepared:
        with self.lock:
            return self._built().prepare()

    def prepare_ranker(self) -> ModelsPrepared:
        with self.lock:
            return self._built().prepare_ranker()

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        with self.lock:
            return self._built().encode(texts, query=query)

    def rerank(self, query: str, texts: Sequence[str]) -> Floats:
        with self.lock:
            return self._built().rerank(query, texts)

    def release_idle(self) -> bool:
        with self.lock:
            return self.built is not None and self.built.release_idle()

    def close(self) -> None:
        with self.lock:
            self.built = None


def deferred(model: str, reranker: str, gpus: str | None) -> Deferred:
    return Deferred(model, reranker, gpus)


class OcrModule(Protocol):
    def ocr_device(self, gpus: str | None) -> "Device": ...

    def loaded_ocr(self, device: "Device") -> "Ocr": ...

    def pages_text(self, ocr: "Ocr", pictures: Sequence[bytes]) -> list[str]: ...

    def let_go(self, device: "Device") -> None: ...


def ocr_module() -> OcrModule:
    return _models_module(OCR, OcrModule)


if TYPE_CHECKING:
    from sift.inference import local as _local
    from sift.inference import ocr as _ocr

    _LOCAL: LocalLoader = _local
    _OCR: OcrModule = _ocr
