from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Help:
    text: str


@dataclass(frozen=True, slots=True)
class Place:
    pass


@dataclass(frozen=True, slots=True)
class Short:
    letter: str


@dataclass(frozen=True, slots=True)
class Read:
    each: Callable[[str], tuple[object, ...]]
