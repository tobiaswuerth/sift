from dataclasses import dataclass, field

import pynvml
import pytest
import torch

from sift.inference.accelerator import CPU, GIB, Device, budget, chosen, threads_for, too_large


@dataclass(slots=True)
class FakeCuda:
    free: int
    total: int
    used: int | None = None
    available: bool = True
    bf16: bool = True
    fractions: list[float] = field(default_factory=list[float])

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16

    def mem_get_info(self, device: int) -> tuple[int, int]:
        assert device == 0
        return self.free, self.total

    def device_memory_used(self, device: int) -> int:
        assert device == 0
        if self.used is None:
            raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_SUPPORTED)
        return self.used

    def set_per_process_memory_fraction(self, fraction: float, device: int) -> None:
        assert device == 0
        self.fractions.append(fraction)


def test_the_budget_is_what_others_leave_free_less_a_tenth_or_a_gib() -> None:
    assert budget(20 * GIB, 24 * GIB) == round(20 * GIB - 2.4 * GIB)
    assert budget(4 * GIB, 8 * GIB) == 3 * GIB
    assert budget(int(2.9 * GIB), 8 * GIB) is None


def test_on_a_gpu_with_room_sift_caps_itself_at_its_budget_in_bfloat16() -> None:
    cuda = FakeCuda(free=20 * GIB, total=24 * GIB, used=6 * GIB)
    assert chosen(cuda, mps=False) == Device("cuda", torch.bfloat16)
    assert cuda.fractions == [pytest.approx((18 - 2.4) / 24)]
    assert chosen(FakeCuda(free=20 * GIB, total=24 * GIB, bf16=False), mps=False).dtype == torch.float32


def test_a_gpu_others_fill_leaves_sift_on_the_cpu() -> None:
    cuda = FakeCuda(free=20 * GIB, total=24 * GIB, used=22 * GIB)
    assert chosen(cuda, mps=False) == CPU
    assert cuda.fractions == []


def test_without_a_gpu_an_apple_one_or_the_cpu_is_chosen() -> None:
    assert chosen(FakeCuda(0, 0, available=False), mps=True) == Device("mps", torch.float32)
    assert chosen(FakeCuda(0, 0, available=False), mps=False) == CPU


def test_beside_a_gpu_pytorch_keeps_one_thread_and_on_the_cpu_one_a_core() -> None:
    assert threads_for(Device("cuda", torch.bfloat16), 32) == 1
    assert threads_for(CPU, 32) == 32


def test_out_of_memory_is_told_from_other_failures() -> None:
    assert too_large(torch.OutOfMemoryError("CUDA out of memory"))
    assert too_large(RuntimeError("MPS backend out of memory (MPS allocated: 9 GiB)"))
    assert not too_large(RuntimeError("shapes do not match"))
