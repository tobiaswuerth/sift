import errno
from dataclasses import dataclass, field

import pynvml
import pytest
import torch

from sift.inference.accelerator import CACHES, CPU, CUDA_FIX, GIB, Device, Found, budget, chosen, emptied, too_large

NONE = Found(mps=False, nvidia=False, cuda_build=None)


@dataclass(frozen=True, slots=True)
class Properties:
    total_memory: int


@dataclass(slots=True)
class FakeCuda:
    free: int
    total: int
    used: int | None = None
    available: bool = True
    bf16: bool = True
    failure: Exception | None = None
    fractions: list[float] = field(default_factory=list[float])
    emulated: list[bool] = field(default_factory=list[bool])
    contexts: list[int] = field(default_factory=list[int])

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self, *, including_emulation: bool = True) -> bool:
        self.emulated.append(including_emulation)
        return self.bf16

    def mem_get_info(self, device: int) -> tuple[int, int]:
        self.contexts.append(device)
        return self.free, self.total

    def device_memory_used(self, device: int) -> int:
        assert device == 0
        if self.failure is not None:
            raise self.failure
        if self.used is None:
            raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_SUPPORTED)
        return self.used

    def get_device_properties(self, device: int) -> Properties:
        assert device == 0
        return Properties(self.total)

    def set_per_process_memory_fraction(self, fraction: float, device: int) -> None:
        assert device == 0
        self.fractions.append(fraction)


def test_the_budget_is_what_others_leave_free_less_a_tenth_or_a_gib() -> None:
    assert budget(20 * GIB, 24 * GIB) == round(20 * GIB - 2.4 * GIB)
    assert budget(4 * GIB, 8 * GIB) == 3 * GIB
    assert budget(int(2.9 * GIB), 8 * GIB) is None


def test_on_a_gpu_with_room_sift_caps_itself_at_its_budget_in_native_bfloat16() -> None:
    cuda = FakeCuda(free=20 * GIB, total=24 * GIB, used=6 * GIB)
    assert chosen(cuda, NONE) == Device("cuda", torch.bfloat16, threads=1)
    assert (cuda.fractions, cuda.contexts) == ([pytest.approx((18 - 2.4) / 24)], [])
    assert cuda.emulated == [False]
    assert chosen(FakeCuda(free=20 * GIB, total=24 * GIB, used=0, bf16=False), NONE).dtype == torch.float32


@pytest.mark.parametrize("failure", [RuntimeError("driver not loaded"), FileNotFoundError("libnvidia-ml.so.1")])
def test_without_nvml_cuda_s_own_figure_of_free_memory_counts(failure: Exception) -> None:
    cuda = FakeCuda(free=20 * GIB, total=24 * GIB, failure=failure)
    assert (chosen(cuda, NONE).name, cuda.contexts) == ("cuda", [0])


def test_a_gpu_others_fill_leaves_sift_on_the_cpu_saying_so_and_holding_no_context_on_it() -> None:
    cuda = FakeCuda(free=20 * GIB, total=24 * GIB, used=22 * GIB)
    found = chosen(cuda, NONE)
    assert (found.name, cuda.fractions, cuda.contexts) == ("cpu", [], [])
    assert found.notice == "only 2.0 GiB of the GPU's memory is free, too little: the models run on the CPU"


def test_without_a_cuda_gpu_an_apple_one_or_the_cpu_is_chosen_saying_why_an_nvidia_one_sits_unused() -> None:
    absent = FakeCuda(0, 0, available=False)
    assert chosen(absent, Found(mps=True, nvidia=False, cuda_build=None)) == Device("mps", torch.float32)
    assert chosen(absent, NONE) == CPU
    unbuilt = chosen(absent, Found(mps=False, nvidia=True, cuda_build=None)).notice
    assert unbuilt is not None
    assert unbuilt.endswith(CUDA_FIX)
    hidden = chosen(absent, Found(mps=False, nvidia=True, cuda_build="12.6")).notice
    assert hidden is not None
    assert "CUDA_VISIBLE_DEVICES" in hidden


class LocalizedError(OSError):
    winerror: int = 14


def test_out_of_memory_is_told_from_other_failures() -> None:
    assert too_large(torch.OutOfMemoryError("CUDA out of memory"))
    assert too_large(RuntimeError("MPS backend out of memory (MPS allocated: 9 GiB)"))
    assert too_large(RuntimeError("CUDA error: out of memory"))
    assert too_large(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"))
    assert too_large(RuntimeError("DefaultCPUAllocator: not enough memory: you tried to allocate 8 GiB"))
    assert too_large(OSError("[WinError 1455] The paging file is too small for this operation to complete"))
    assert too_large(MemoryError())
    assert too_large(OSError(errno.ENOMEM, "Cannot allocate memory"))
    assert too_large(OSError("[WinError 8] Not enough memory resources are available to process this command"))
    assert too_large(LocalizedError("Nicht genügend Arbeitsspeicher verfügbar"))
    assert not too_large(OSError(errno.ENOENT, "No such file or directory"))
    assert not too_large(OSError(errno.ENOSPC, "No space left on device"))
    assert too_large(RuntimeError("Out of memory while allocating a tensor"))
    assert not too_large(RuntimeError("shapes do not match"))


def test_after_running_out_of_memory_the_cache_of_the_device_alone_is_emptied() -> None:
    emptied_on: list[str] = []
    caches = {"cuda": lambda: emptied_on.append("cuda"), "mps": lambda: emptied_on.append("mps")}
    emptied(Device("cuda", torch.bfloat16), caches)
    emptied(CPU, caches)
    assert emptied_on == ["cuda"]
    assert (CACHES["cuda"], CACHES["mps"]) == (torch.cuda.empty_cache, torch.mps.empty_cache)
