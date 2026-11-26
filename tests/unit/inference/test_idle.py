from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import cast

import numpy as np
import pytest
import torch

from sift.api.errors import ModelUnavailableError
from sift.inference.accelerator import BATCH, CPU_THREADS, GIB, GPU_THREADS, Device, Machine
from sift.inference.held import RERANK_BATCH, Embedder, Loaded, Ranker
from sift.inference.local import FULL_REMEDIES, IDLE_SECONDS, Local, LocalHost
from sift.inference.prepared import ModelsPrepared
from tests.devices import FakeCuda
from tests.unit.inference.states import placed


@dataclass(eq=False)
class Movable:
    on: str
    dtype: torch.dtype
    moves: list[str] = field(default_factory=list[str])
    asked: list[tuple[object, ...]] = field(default_factory=list[tuple[object, ...]])
    max_seq_length: int | None = 512
    tokenizer: object = None
    refusals: dict[str, BaseException] = field(default_factory=dict[str, BaseException])
    cuda: FakeCuda | None = None
    size: int = 0
    most_batch: int | None = None

    def to(self, device: str) -> "Movable":
        if device in self.refusals:
            raise self.refusals.pop(device)
        self.moves.append(device)
        self.on = device
        self.held()
        return self

    def held(self) -> None:
        if self.cuda is not None and self.on == "cuda":
            self.cuda.allocate(self.size)

    def modules(self) -> Iterator[object]:
        return iter(())

    def get_embedding_dimension(self) -> int:
        return 4

    def encode_query(
        self, inputs: list[str], *, batch_size: int, normalize_embeddings: bool, show_progress_bar: bool
    ) -> object:
        self.asked.append((self.on, inputs, batch_size, normalize_embeddings, show_progress_bar))
        return np.ones((len(inputs), 4), np.float32) / 2

    def encode_document(
        self, inputs: list[str], *, batch_size: int, normalize_embeddings: bool, show_progress_bar: bool
    ) -> object:
        return self.encode_query(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )

    def predict(
        self, inputs: list[tuple[str, str]], *, batch_size: int, show_progress_bar: bool, activation_fn: torch.nn.Module
    ) -> object:
        self.asked.append((self.on, inputs, batch_size, show_progress_bar, activation_fn))
        if self.most_batch is not None and batch_size > self.most_batch:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return np.zeros(len(inputs), np.float32)


@dataclass(slots=True)
class Room:
    now: float = 0.0
    cuda: FakeCuda = field(default_factory=lambda: FakeCuda(free=20 * GIB, total=24 * GIB))
    threads: list[int] = field(default_factory=list[int])
    loaded: list[Movable] = field(default_factory=list[Movable])
    emptied: list[str] = field(default_factory=list[str])
    model_size: int = 0

    def _made(self, device: Device) -> Movable:
        made = Movable(device.name, device.dtype, cuda=self.cuda, size=self.model_size)
        made.held()
        self.loaded.append(made)
        return made

    def embedder(self, _name: str, device: Device) -> Loaded:
        return Loaded(cast("Embedder", self._made(device)), 4)

    def ranker(self, _name: str, device: Device) -> Ranker:
        return cast("Ranker", self._made(device))

    def _emptied(self) -> None:
        self.emptied.append("cuda")
        self.cuda.empty_cache()

    def local(self, *, gpu: bool = True) -> Local:
        self.cuda.available = gpu
        machine = Machine(self.cuda, lambda: False, lambda: gpu, "13.0", self.threads.append)
        caches = {"cuda": self._emptied}
        host = LocalHost(machine, lambda: self.now, self.embedder, self.ranker, caches)
        return Local("an-embedder", "a-reranker", host=host)


def used(local: Local) -> None:
    local.encode(["a question"], query=True)
    local.rerank("a question", ["a passage"])


def test_idle_models_leave_the_gpu_for_ram_and_go_back_when_used() -> None:
    room = Room()
    local = room.local()
    used(local)
    room.now = IDLE_SECONDS - 0.5
    assert not local.release_idle()
    room.now = IDLE_SECONDS
    assert local.release_idle()
    assert ([model.on for model in room.loaded], room.emptied, local.release_idle()) == (
        ["cpu", "cpu"],
        ["cuda"],
        False,
    )
    room.cuda.free = 16 * GIB
    used(local)
    assert [(model.on, model.moves) for model in room.loaded] == [("cuda", ["cpu", "cuda"])] * 2
    assert [[asked[0] for asked in model.asked] for model in room.loaded] == [["cuda", "cuda"]] * 2
    assert room.cuda.fractions == pytest.approx([(20 - 2.4) / 24, (16 - 2.4) / 24])


def test_the_models_own_memory_is_freed_on_rest_and_counted_again_on_return() -> None:
    room = Room(model_size=2 * GIB)
    local = room.local()
    used(local)
    assert (room.cuda.free, room.cuda.reserved) == (16 * GIB, 4 * GIB)
    room.now = IDLE_SECONDS
    assert local.release_idle() is True
    assert (room.cuda.free, room.cuda.reserved) == (20 * GIB, 0)
    used(local)
    assert (room.cuda.free, room.cuda.reserved) == (16 * GIB, 4 * GIB)
    assert room.cuda.fractions == pytest.approx([(20 - 2.4) / 24] * 2)


def test_every_use_starts_the_idle_time_again() -> None:
    room = Room()
    local = room.local()
    used(local)
    room.now = IDLE_SECONDS * 0.75
    local.encode(["again"], query=True)
    room.now = IDLE_SECONDS * 1.5
    assert not local.release_idle()
    room.now = IDLE_SECONDS * 1.75
    assert local.release_idle()


def test_nothing_moves_on_the_cpu() -> None:
    room = Room()
    local = room.local(gpu=False)
    used(local)
    room.now = IDLE_SECONDS * 10
    assert (local.release_idle(), [model.moves for model in room.loaded], room.emptied) == (False, [[], []], [])


def test_back_from_rest_the_budget_is_measured_again_and_a_busy_gpu_means_the_models_reload_for_the_cpu() -> None:
    room = Room()
    local = room.local()
    used(local)
    local.state = replace(placed(local), batch=3, rerank_batch=2)
    room.now = IDLE_SECONDS
    assert local.release_idle()
    room.cuda.free = GIB
    used(local)
    assert (placed(local).batch, placed(local).rerank_batch) == (BATCH, RERANK_BATCH)
    on_cpu = [(model.on, model.dtype) for model in room.loaded[2:]]
    assert (on_cpu, room.threads) == ([("cpu", torch.float32)] * 2, [GPU_THREADS] * 2 + [CPU_THREADS] * 2)
    assert local.prepare().notices == (
        "only 1.0 GiB of the GPU's memory is free, too little: the models run on the CPU",
    )
    assert len(room.cuda.fractions) == 1


def test_models_on_the_cpu_because_the_gpu_was_busy_go_to_the_gpu_once_it_has_room() -> None:
    room = Room()
    room.cuda.free = GIB
    local = room.local()
    used(local)
    assert [(model.on, model.dtype) for model in room.loaded] == [("cpu", torch.float32)] * 2
    room.now = IDLE_SECONDS - 0.5
    assert not local.release_idle()
    room.now = IDLE_SECONDS
    assert (local.release_idle(), room.emptied) == (True, [])
    room.cuda.free = 20 * GIB
    used(local)
    assert [(model.on, model.dtype) for model in room.loaded[2:]] == [("cuda", torch.bfloat16)] * 2
    assert local.prepare() == ModelsPrepared((), built=False)


def test_a_move_back_to_the_gpu_that_runs_out_of_memory_leaves_the_models_resting() -> None:
    room = Room()
    local = room.local()
    used(local)
    room.now = IDLE_SECONDS
    assert local.release_idle()
    room.loaded[1].refusals["cuda"] = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    with pytest.raises(ModelUnavailableError) as refused:
        used(local)
    assert refused.value.remedy == FULL_REMEDIES["cuda"]
    assert ([model.on for model in room.loaded], room.emptied) == (["cpu", "cpu"], ["cuda", "cuda"])
    used(local)
    assert [model.on for model in room.loaded] == ["cuda", "cuda"]


def test_models_that_cannot_move_to_ram_stay_on_the_gpu_and_rest_the_next_time() -> None:
    room = Room()
    local = room.local()
    used(local)
    room.loaded[1].refusals["cpu"] = RuntimeError("DefaultCPUAllocator: can't allocate memory")
    room.now = IDLE_SECONDS
    with pytest.raises(RuntimeError, match="can't allocate memory"):
        local.release_idle()
    assert ([model.on for model in room.loaded], room.emptied) == (["cuda", "cuda"], [])
    assert local.release_idle()
    assert ([model.on for model in room.loaded], room.emptied) == (["cpu", "cpu"], ["cuda"])


def test_a_model_never_loaded_is_not_moved_when_the_others_rest() -> None:
    room = Room()
    local = room.local()
    local.encode(["a question"], query=True)
    room.now = IDLE_SECONDS
    assert (local.release_idle(), [model.moves for model in room.loaded]) == (True, [["cpu"]])


def predicted(room: Room) -> list[tuple[int, int]]:
    asked = cast("list[tuple[str, list[tuple[str, str]], int, bool, object]]", room.loaded[0].asked)
    return [(len(inputs), batch) for _, inputs, batch, _, _ in asked]


def test_the_reranker_scores_all_its_passages_in_one_call_longest_first_a_batch_of_sixteen_at_a_time() -> None:
    room = Room()
    local = room.local()
    texts = ["passage " * (1 + n % 7) for n in range(40)]
    assert local.rerank("a question", texts).shape == (40,)
    assert predicted(room) == [(40, RERANK_BATCH)]
    ((_, inputs, *_),) = cast("list[tuple[str, list[tuple[str, str]], int, bool, object]]", room.loaded[0].asked)
    assert [len(text) for _, text in inputs] == sorted(map(len, texts), reverse=True)
    assert local.rerank("a question", texts[:3]).shape == (3,)
    assert predicted(room)[1:] == [(3, 3)]
    assert placed(local).rerank_batch == RERANK_BATCH


def test_a_reranker_short_of_memory_scores_fewer_at_a_time_from_then_on() -> None:
    room = Room()
    local = room.local()
    local.prepare_ranker()
    room.loaded[0].most_batch = 4
    assert local.rerank("a question", ["passage"] * 40).shape == (40,)
    refused = [(40, 16), (20, 16), (10, 10), (5, 5)]
    assert predicted(room) == [*refused, *[(2, 2)] * 20]
    assert placed(local).rerank_batch == 2
    assert local.rerank("a question", ["passage"] * 9).shape == (9,)
    assert predicted(room)[-1] == (9, 2)
