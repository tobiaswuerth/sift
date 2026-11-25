import logging
import threading
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from sift.domain.codec.codes import Floats, Halves
from sift.inference import local, ocr
from sift.inference.hub import pinned
from sift.inference.loading import REDIRECTS, Deferred, LocalLoader, OcrModule, Resting, ocr_module
from sift.inference.prepared import ModelsPrepared
from tests.models import Hashed


@dataclass(slots=True)
class Crowded:
    arrived: threading.Barrier
    builds: list[tuple[str, str, str | None]] = field(default_factory=list[tuple[str, str, str | None]])
    calls: int = 0
    at_once: int = 0
    most_at_once: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def local(self, model: str, reranker: str, gpus: str | None) -> Resting:
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

    def prepare(self) -> ModelsPrepared:
        return self.inner.prepare()

    def prepare_ranker(self) -> ModelsPrepared:
        return self.inner.prepare_ranker()

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        self.crowded.entered()
        try:
            return self.inner.encode(texts, query=query)
        finally:
            self.crowded.left()

    def rerank(self, query: str, texts: Sequence[str]) -> Floats:
        return self.inner.rerank(query, texts)

    def release_idle(self) -> bool:
        self.crowded.entered()
        self.crowded.left()
        return True


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


def test_idle_models_are_let_go_one_call_at_a_time_and_none_are_built_to_let_go_of() -> None:
    crowded = Crowded(threading.Barrier(8))
    shared = Deferred("model", "reranker", None, loader=lambda: crowded)
    assert (shared.release_idle(), crowded.builds) == (False, [])
    shared.encode(["dark matter"], query=True)

    def use(n: int) -> None:
        crowded.arrived.wait(30)
        for _ in range(20):
            assert shared.release_idle() if n % 2 else len(shared.encode(["halo"], query=True)) == 1

    threads = [threading.Thread(target=use, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert (crowded.calls, crowded.most_at_once) == (161, 1)


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


def test_models_keep_the_name_they_were_first_named_by_while_their_folder_changes(tmp_path: Path) -> None:
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_bytes(b"{}")
    shared = Deferred(str(folder), "reranker", None, loader=lambda: Crowded(threading.Barrier(1)))
    named = shared.name
    (folder / "config.json").write_bytes(b'{"changed": true}')
    assert (shared.name, pinned(str(folder)) != named) == (named, True)
