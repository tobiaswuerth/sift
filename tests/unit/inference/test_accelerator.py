import errno
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pynvml
import pytest
import torch

from sift.inference.accelerator import (
    CACHES,
    CPU,
    CPU_THREADS,
    CUDA_FIX,
    GIB,
    Device,
    Hardware,
    Machine,
    budget,
    chosen,
    emptied,
    nvidia,
    placed,
    too_large,
)

NONE = Hardware(mps=False, nvidia=False, cuda_build=None)


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


def test_a_recursion_too_deep_while_memory_is_looked_at_is_no_missing_nvml() -> None:
    with pytest.raises(RecursionError):
        chosen(FakeCuda(free=20 * GIB, total=24 * GIB, failure=RecursionError("too deep")), NONE)


def test_a_gpu_others_fill_leaves_sift_on_the_cpu_saying_so_and_holding_no_context_on_it() -> None:
    cuda = FakeCuda(free=20 * GIB, total=24 * GIB, used=22 * GIB)
    found = chosen(cuda, NONE)
    assert (found.name, cuda.fractions, cuda.contexts) == ("cpu", [], [])
    assert found.notice == "only 2.0 GiB of the GPU's memory is free, too little: the models run on the CPU"


def test_without_a_cuda_gpu_an_apple_one_or_the_cpu_is_chosen_saying_why_an_nvidia_one_sits_unused() -> None:
    absent = FakeCuda(0, 0, available=False)
    assert chosen(absent, Hardware(mps=True, nvidia=False, cuda_build=None)) == Device("mps", torch.float32)
    assert chosen(absent, NONE) == CPU
    unbuilt = chosen(absent, Hardware(mps=False, nvidia=True, cuda_build=None)).notice
    assert unbuilt is not None
    assert unbuilt.endswith(CUDA_FIX)
    unseen = chosen(absent, Hardware(mps=False, nvidia=True, cuda_build="12.6")).notice
    assert unseen == (
        "PyTorch (CUDA 12.6) can use no NVIDIA GPU here: the models run on the CPU; update the NVIDIA driver"
    )


@pytest.mark.parametrize("gpus", ["-1", ""])
def test_a_gpu_hidden_on_purpose_is_no_driver_to_update(gpus: str) -> None:
    hidden = chosen(FakeCuda(0, 0, available=False), Hardware(mps=False, nvidia=True, cuda_build="12.6", gpus=gpus))
    said = f"PyTorch sees no NVIDIA GPU with CUDA_VISIBLE_DEVICES set to {gpus!r}: the models run on the CPU"
    assert hidden.notice == said
    unbuilt = chosen(FakeCuda(0, 0, available=False), Hardware(mps=False, nvidia=True, cuda_build=None, gpus=gpus))
    assert unbuilt.notice is not None
    assert unbuilt.notice.endswith(CUDA_FIX)


def unseen_mps() -> bool:
    return False


def seen_mps() -> bool:
    return True


def machine(cuda: FakeCuda, held: list[int], mps: Callable[[], bool] = unseen_mps) -> Machine:
    return Machine(cuda=cuda, mps=mps, nvidia=lambda: False, cuda_build="12.6", threads=held.append)


@pytest.mark.parametrize(
    "cuda, mps, threads",
    [
        (FakeCuda(free=20 * GIB, total=24 * GIB, used=0), unseen_mps, 1),
        (FakeCuda(free=20 * GIB, total=24 * GIB, used=22 * GIB), unseen_mps, CPU_THREADS),
        (FakeCuda(0, 0, available=False), seen_mps, CPU_THREADS),
        (FakeCuda(0, 0, available=False), unseen_mps, CPU_THREADS),
    ],
)
def test_pytorch_keeps_one_cpu_thread_beside_an_nvidia_gpu_and_one_a_core_elsewhere(
    cuda: FakeCuda, mps: Callable[[], bool], threads: int
) -> None:
    held: list[int] = []
    device = placed(machine(cuda, held, mps), None)
    assert (device.threads, held) == (threads, [threads])


def virtual_mac() -> bool:
    warnings.warn("Skipping device Apple Paravirtual device that does not support Metal 2.0", UserWarning, stacklevel=2)
    return False


def warning_mac() -> bool:
    warnings.warn("MPS is deprecated", UserWarning, stacklevel=2)
    return False


def test_a_virtual_mac_gpu_is_passed_over_quietly_and_other_warnings_still_heard() -> None:
    assert placed(machine(FakeCuda(0, 0, available=False), [], virtual_mac), None) == CPU
    with pytest.warns(UserWarning, match="MPS is deprecated"):
        placed(machine(FakeCuda(0, 0, available=False), [], warning_mac), None)


def test_an_nvidia_gpu_is_known_by_the_nvidia_smi_its_driver_brings(tmp_path: Path) -> None:
    assert not nvidia(str(tmp_path))
    for name in ("nvidia-smi", "nvidia-smi.exe"):
        (tmp_path / name).write_bytes(b"")
        (tmp_path / name).chmod(0o755)
    assert nvidia(str(tmp_path))


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
