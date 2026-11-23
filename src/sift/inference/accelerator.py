from dataclasses import dataclass
from typing import Protocol

import pynvml
import torch

GIB = 1 << 30
HEADROOM = 0.1
LEAST_BUDGET = 2 * GIB
BATCH = 64
GPU_THREADS = 1


class Cuda(Protocol):
    def is_available(self) -> bool: ...

    def is_bf16_supported(self) -> bool: ...

    def mem_get_info(self, device: int) -> tuple[int, int]: ...

    def device_memory_used(self, device: int) -> int: ...

    def set_per_process_memory_fraction(self, fraction: float, device: int) -> None: ...


@dataclass(frozen=True, slots=True)
class Device:
    name: str
    dtype: torch.dtype
    batch: int = BATCH


CPU = Device("cpu", torch.float32)


def budget(free: int, total: int) -> int | None:
    held = free - max(round(HEADROOM * total), GIB)
    return held if held >= LEAST_BUDGET else None


def _free(cuda: Cuda) -> tuple[int, int]:
    free, total = cuda.mem_get_info(0)
    try:
        used = cuda.device_memory_used(0)
    except (ModuleNotFoundError, pynvml.NVMLError):
        return free, total
    return min(free, total - used), total


def _on_cuda(cuda: Cuda) -> Device | None:
    free, total = _free(cuda)
    held = budget(free, total)
    if held is None:
        return None
    cuda.set_per_process_memory_fraction(held / total, 0)
    return Device("cuda", torch.bfloat16 if cuda.is_bf16_supported() else torch.float32)


def chosen(cuda: Cuda, *, mps: bool) -> Device:
    found = _on_cuda(cuda) if cuda.is_available() else None
    if found is not None:
        return found
    return Device("mps", torch.float32) if mps else CPU


def threads_for(device: Device, cores: int) -> int:
    return GPU_THREADS if device.name == "cuda" else cores


def too_large(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "MPS backend out of memory" in str(error)
