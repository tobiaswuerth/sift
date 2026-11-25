import errno
from collections.abc import Callable
from pathlib import Path

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
from tests.devices import FakeCuda

NONE = Hardware(mps=False, nvidia=False, cuda_build=None)


def test_the_budget_is_what_others_leave_free_less_a_tenth_or_a_gib() -> None:
    assert budget(20 * GIB, 24 * GIB) == round(20 * GIB - 2.4 * GIB)
    assert budget(4 * GIB, 8 * GIB) == 3 * GIB
    assert budget(int(2.9 * GIB), 8 * GIB) is None


def test_sift_s_cap_counts_the_memory_it_already_holds_as_room() -> None:
    cuda = FakeCuda(free=10 * GIB, total=24 * GIB, used=14 * GIB, reserved=4 * GIB)
    assert chosen(cuda, NONE).name == "cuda"
    assert cuda.fractions == [pytest.approx((10 + 4 - 2.4) / 24)]
    held = FakeCuda(free=2 * GIB, total=24 * GIB, used=22 * GIB, reserved=6 * GIB)
    assert (chosen(held, NONE).name, held.fractions) == ("cuda", [pytest.approx((2 + 6 - 2.4) / 24)])


def test_on_a_gpu_with_room_sift_caps_itself_at_its_budget_in_native_bfloat16() -> None:
    cuda = FakeCuda(free=20 * GIB, total=24 * GIB, used=6 * GIB)
    assert chosen(cuda, NONE) == Device("cuda", torch.bfloat16, threads=1)
    assert (cuda.fractions, cuda.contexts) == ([pytest.approx((18 - 2.4) / 24)], [])
    beside_mps = FakeCuda(free=20 * GIB, total=24 * GIB, used=6 * GIB)
    assert chosen(beside_mps, Hardware(mps=True, nvidia=True, cuda_build="12.6")) == Device(
        "cuda", torch.bfloat16, threads=1
    )
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
    assert (device.threads, held) == (threads, [])


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
