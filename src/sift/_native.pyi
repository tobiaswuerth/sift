from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import NDArray

type ListsRow = tuple[int, str, int, bytes | None, bytes | None, bytes | None]
type ListBlockRow = tuple[int, str, int, bytes, bytes]
type Listed = tuple[list[ListsRow], list[ListBlockRow]]

class Words:
    def __new__(cls, stored: bytes, /) -> Self: ...
    def building(self, most: int, most_field: int, /) -> Building: ...

class Building:
    @property
    def chunks(self) -> int: ...
    @property
    def tokens(self) -> int: ...
    def add(
        self, places: NDArray[np.int64], of: Sequence[int], fields: Sequence[tuple[str, str]], texts: Sequence[str], /
    ) -> int: ...
    @property
    def first(self) -> int | None: ...
    @property
    def last(self) -> int | None: ...
    def written(
        self, segment: int, average: float, shape: tuple[int, int, int, int], /
    ) -> tuple[Listed, list[tuple[int, int, bytes]], list[tuple[str, int]]]: ...

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
    def step(self, segment: int, /) -> tuple[Listed, list[int], bool]: ...

def blocked_all(each: Sequence[bytes], /) -> tuple[bytes, bytes, list[int]]: ...
def with_lengths(ids: NDArray[np.int64], lengths: NDArray[np.int64], shift: int, /) -> list[tuple[int, bytes]]: ...
def positions_all(each: Sequence[bytes], counts: NDArray[np.int64], /) -> bytes: ...
