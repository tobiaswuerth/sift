from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import NDArray

type Blocks = list[tuple[bytes, bytes]]
type ListsRow = tuple[str, int, bytes | None, bytes | None, bytes | None, Blocks]

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
    def written(
        self, average: float, range_shift: int, block: int, long: int, lengths_shift: int, /
    ) -> tuple[int, int, list[ListsRow], list[tuple[int, bytes]]]: ...

def blocked_all(each: Sequence[bytes], /) -> tuple[bytes, bytes, list[int]]: ...
def positions_all(each: Sequence[bytes], counts: NDArray[np.int64], /) -> bytes: ...
def encoded(
    ids: NDArray[np.int64], counts: NDArray[np.int64], positions: NDArray[np.int64], bounds: Sequence[int], /
) -> tuple[list[bytes], list[bytes]]: ...
