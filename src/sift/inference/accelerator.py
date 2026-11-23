import errno
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import pynvml
import torch

GIB = 1 << 30
HEADROOM = 0.1
LEAST_BUDGET = 2 * GIB
BATCH = 64
GPU_THREADS = 1
CPU_THREADS = int(torch.get_num_threads())
DEVICE_FULL = ("out of memory", "cublas_status_alloc_failed")
HOST_FULL = ("not enough memory", "paging file is too small")
HOST_FULL_WINDOWS = frozenset({8, 14, 1455})
CUDA_FIX = "uv tool install --reinstall --torch-backend auto sift-rag"
UNBUILT = f"an NVIDIA GPU is there, but this PyTorch has no CUDA: the models run on the CPU; to use it: {CUDA_FIX}"
HIDDEN = "PyTorch can use no NVIDIA GPU here (see CUDA_VISIBLE_DEVICES and the driver): the models run on the CPU"


class Properties(Protocol):
    @property
    def total_memory(self) -> int: ...


class Cuda(Protocol):
    def is_available(self) -> bool: ...

    def is_bf16_supported(self, *, including_emulation: bool = ...) -> bool: ...

    def mem_get_info(self, device: int) -> tuple[int, int]: ...

    def device_memory_used(self, device: int) -> int: ...

    def get_device_properties(self, device: int) -> Properties: ...

    def set_per_process_memory_fraction(self, fraction: float, device: int) -> None: ...


@dataclass(frozen=True, slots=True)
class Hardware:
    mps: bool
    nvidia: bool
    cuda_build: str | None


@dataclass(frozen=True, slots=True)
class Device:
    name: str
    dtype: torch.dtype
    batch: int = BATCH
    threads: int = CPU_THREADS
    notice: str | None = None


CPU = Device("cpu", torch.float32)
CACHES: Mapping[str, Callable[[], None]] = MappingProxyType(
    {"cuda": torch.cuda.empty_cache, "mps": torch.mps.empty_cache}
)


def budget(free: int, total: int) -> int | None:
    held = free - max(round(HEADROOM * total), GIB)
    return held if held >= LEAST_BUDGET else None


def _free(cuda: Cuda) -> tuple[int, int]:
    total = cuda.get_device_properties(0).total_memory
    try:
        used = cuda.device_memory_used(0)
    except (ModuleNotFoundError, RuntimeError, OSError, pynvml.NVMLError):
        return cuda.mem_get_info(0)
    return total - used, total


def _on_cuda(cuda: Cuda) -> Device:
    free, total = _free(cuda)
    held = budget(free, total)
    if held is None:
        notice = f"only {free / GIB:.1f} GiB of the GPU's memory is free, too little: the models run on the CPU"
        return Device("cpu", torch.float32, notice=notice)
    cuda.set_per_process_memory_fraction(held / total, 0)
    dtype = torch.bfloat16 if cuda.is_bf16_supported(including_emulation=False) else torch.float32
    return Device("cuda", dtype, threads=GPU_THREADS)


def _unused(found: Hardware) -> str | None:
    if not found.nvidia:
        return None
    return UNBUILT if found.cuda_build is None else HIDDEN


def chosen(cuda: Cuda, found: Hardware) -> Device:
    if cuda.is_available():
        return _on_cuda(cuda)
    if found.mps:
        return Device("mps", torch.float32)
    notice = _unused(found)
    return CPU if notice is None else Device("cpu", torch.float32, notice=notice)


def nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def _host_full_code(error: BaseException) -> bool:
    code: object = getattr(error, "winerror", None)
    return isinstance(error, OSError) and (error.errno == errno.ENOMEM or code in HOST_FULL_WINDOWS)


def host_full(error: BaseException) -> bool:
    said = str(error).lower()
    return isinstance(error, MemoryError) or _host_full_code(error) or any(sign in said for sign in HOST_FULL)


def too_large(error: BaseException) -> bool:
    said = str(error).lower()
    device_full = isinstance(error, torch.OutOfMemoryError) or any(sign in said for sign in DEVICE_FULL)
    return device_full or host_full(error)


def emptied(device: Device, caches: Mapping[str, Callable[[], None]] = CACHES) -> None:
    empty = caches.get(device.name)
    if empty is not None:
        empty()
