from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Help:
    text: str


@dataclass(frozen=True, slots=True)
class AbsolutePath:
    pass


@dataclass(frozen=True, slots=True)
class Composed:
    pass


@dataclass(frozen=True, slots=True)
class Bounds:
    low: int
    high: int
