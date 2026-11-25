from dataclasses import dataclass, field

import pynvml


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
    reserved: int = 0
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

    def memory_reserved(self, device: int) -> int:
        assert device == 0
        return self.reserved

    def get_device_properties(self, device: int) -> Properties:
        assert device == 0
        return Properties(self.total)

    def set_per_process_memory_fraction(self, fraction: float, device: int) -> None:
        assert device == 0
        self.fractions.append(fraction)

    def allocate(self, size: int) -> None:
        self.free, self.reserved = self.free - size, self.reserved + size

    def empty_cache(self) -> None:
        self.free, self.reserved = self.free + self.reserved, 0
