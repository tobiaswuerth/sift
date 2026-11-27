from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

def blocked_all(each: Sequence[bytes], /) -> tuple[bytes, bytes, list[int]]: ...
def positions_all(each: Sequence[bytes], counts: NDArray[np.int64], /) -> bytes: ...
def encoded(
    ids: NDArray[np.int64], counts: NDArray[np.int64], positions: NDArray[np.int64], bounds: Sequence[int], /
) -> tuple[list[bytes], list[bytes]]: ...
