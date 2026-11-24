import logging
import threading
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from sift.domain.codec.codes import Floats, Halves
from sift.inference import local, ocr
from sift.inference.loading import REDIRECTS, Deferred, LocalLoader, OcrModule, ocr_module
from sift.inference.models import Models
from tests.models import Hashed


@dataclass(slots=True)
class Crowded:
    arrived: threading.Barrier
    builds: list[tuple[str, str, str | None]] = field(default_factory=list[tuple[str, str, str | None]])
    calls: int = 0
    at_once: int = 0
    most_at_once: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def local(self, model: str, reranker: str, gpus: str | None) -> Models:
        with self.lock:
            self.builds.append((model, reranker, gpus))
        return Watched(self)

    def entered(self) -> None:
        with self.lock:
            self.calls, self.at_once = self.calls + 1, self.at_once + 1
            self.most_at_once = max(self.most_at_once, self.at_once)

    def left(self) -> None:
        with self.lock:
            self.at_once -= 1


@dataclass(frozen=True, slots=True)
class Watched:
    crowded: Crowded
    inner: Hashed = field(default_factory=Hashed)

    @property
    def name(self) -> str:
        return self.inner.name

    def prepare(self) -> tuple[str, ...]:
        return self.inner.prepare()

    def prepare_ranker(self) -> tuple[str, ...]:
        return self.inner.prepare_ranker()

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        self.crowded.entered()
        try:
            return self.inner.encode(texts, query=query)
        finally:
            self.crowded.left()

    def rerank(self, query: str, texts: Sequence[str]) -> Floats:
        return self.inner.rerank(query, texts)


def test_models_shared_by_threads_are_built_once_and_used_by_one_call_at_a_time() -> None:
    crowded = Crowded(threading.Barrier(8))
    shared = Deferred("model", "reranker", None, loader=lambda: crowded)

    def use() -> None:
        crowded.arrived.wait(30)
        for _ in range(20):
            shared.encode(["dark matter"], query=True)

    threads = [threading.Thread(target=use) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert (crowded.builds, crowded.calls, crowded.most_at_once) == ([("model", "reranker", None)], 160, 1)


def test_the_note_pytorch_logs_on_a_mac_as_the_models_load_is_silenced() -> None:
    ocr_module()
    assert logging.getLogger(REDIRECTS).getEffectiveLevel() == logging.ERROR


def test_pytorch_s_warning_of_a_virtual_mac_s_gpu_goes_unheard_once_the_models_load_and_others_are_heard() -> None:
    ocr_module()
    warnings.warn("Skipping device Apple Paravirtual device that does not support Metal 2.0", UserWarning, stacklevel=1)
    with pytest.warns(UserWarning, match="MPS is deprecated"):
        warnings.warn("MPS is deprecated", UserWarning, stacklevel=1)


def test_the_modules_loaded_late_are_what_loading_takes_them_for() -> None:
    loader: LocalLoader = local
    reader: OcrModule = ocr
    assert (loader.local, ocr_module()) == (local.local, reader)
