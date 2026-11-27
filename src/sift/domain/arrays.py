from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

STORED_U32 = np.dtype("<u4")
STORED_I64 = np.dtype("<i8")
STORED_HALF = np.dtype("<f2")

type Ids = NDArray[np.int64]
type Mask = NDArray[np.bool_]
type Scores = NDArray[np.float64]


def read_only[T: np.generic](values: NDArray[T]) -> NDArray[T]:
    values.flags.writeable = False
    return values.view()


def ints(values: NDArray[np.integer]) -> list[int]:
    return cast("list[int]", values.tolist())


def floats(values: NDArray[np.floating]) -> list[float]:
    return cast("list[float]", values.tolist())


def largest(values: NDArray[np.integer]) -> int:
    return int(values.max())


def total(values: NDArray[np.integer]) -> int:
    return int(values.sum())


def first(values: NDArray[np.integer]) -> int:
    return ints(values[:1])[0]


def last(values: NDArray[np.integer]) -> int:
    return ints(values[-1:])[0]


def after(first: int, rest: Ids) -> Ids:
    return np.concatenate((np.array([first], np.int64), rest))


def firsts(values: NDArray[np.generic]) -> Mask:
    return np.concatenate((np.ones(min(len(values), 1), np.bool_), values[1:] != values[:-1]))


def run_starts(values: NDArray[np.generic]) -> Ids:
    return np.flatnonzero(firsts(values))


def sums_at(values: Ids, starts: Ids) -> Ids:
    return cast("Ids", np.add.reduceat(values, starts))


def maxima_at(values: Scores, starts: Ids) -> Scores:
    return cast("Scores", np.maximum.reduceat(values, starts))


@dataclass(eq=False, frozen=True, slots=True)
class Ascending:
    values: Ids

    def __post_init__(self) -> None:
        if len(self.values) > 1 and not bool((self.values[1:] >= self.values[:-1]).all()):
            raise ValueError("these values are not ascending: sort them first")


def ascending(values: Ids) -> Ascending:
    return Ascending(np.sort(values))


def among(values: Ids, held: Ascending) -> Mask:
    if not len(held.values):
        return np.zeros(len(values), np.bool_)
    at = np.minimum(np.searchsorted(held.values, values), len(held.values) - 1)
    return cast("Mask", held.values[at] == values)


def distinct(held: Ascending) -> Ascending:
    return Ascending(held.values[firsts(held.values)])


def either(masks: list[Mask]) -> Mask:
    return cast("Mask", np.logical_or.reduce(masks))


def every(masks: list[Mask]) -> Mask:
    return cast("Mask", np.logical_and.reduce(masks))
