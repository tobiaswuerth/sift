from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Help:
    text: str


@dataclass(frozen=True, slots=True)
class Place:
    pass


@dataclass(frozen=True, slots=True)
class Short:
    flag: str
