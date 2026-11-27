from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import NDArray

type PackRow = tuple[int, str, bytes, bytes]
type RangedRow = tuple[int, str, int, bytes]
type ListBlockRow = tuple[int, str, int, bytes, bytes]
type Listed = tuple[list[PackRow], list[RangedRow], list[ListBlockRow]]

class Words:
    def __new__(cls, stored: bytes, /) -> Self: ...
    def building(self, most: int, most_field: int, /) -> Building: ...

class Building:
    def add(
        self, places: NDArray[np.int64], of: Sequence[int], fields: Sequence[tuple[str, str]], texts: Sequence[str], /
    ) -> int: ...
    def written(
        self, segment: int, average: float, shape: tuple[int, int, int, int], /
    ) -> tuple[Listed, list[tuple[int, int, bytes]]]: ...

class Merger:
    def __new__(
        cls,
        live: NDArray[np.int64],
        lengths: NDArray[np.int64],
        dead: NDArray[np.int64],
        average: float,
        shape: tuple[int, int, int, int],
        limits: tuple[int, int, int],
        sources: int,
        /,
    ) -> Self: ...
    def feed(self, source: int, rows: Sequence[tuple[object, ...]], /) -> bool: ...
    def feed_packs(self, source: int, rows: Sequence[tuple[object, ...]], /) -> bool: ...
    def step(self, segment: int, /) -> tuple[Listed, list[int], bool]: ...

def gathered(building: Building, /) -> tuple[int | None, int | None, int, int]: ...
def unpacked(
    lists: bytes, places: bytes | None, terms: Sequence[str], /
) -> list[tuple[str, int, bytes | None, bytes | None]]: ...
def counted(rows: Sequence[bytes], low: str, high: str | None, /) -> list[tuple[str, int]]: ...
def blocked_all(each: Sequence[bytes], /) -> tuple[bytes, bytes, list[int]]: ...
def with_lengths(ids: NDArray[np.int64], lengths: NDArray[np.int64], shift: int, /) -> list[tuple[int, bytes]]: ...
def positions_all(each: Sequence[bytes], counts: NDArray[np.int64], /) -> bytes: ...
