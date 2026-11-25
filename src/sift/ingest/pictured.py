from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Pages:
    pictures: tuple[bytes, ...]
    count: int


@dataclass(frozen=True, slots=True)
class Unpictured:
    problem: str


@dataclass(frozen=True, slots=True)
class ToPicture:
    path: Path
    size: int
    raw: bytes
    most: int


type Pictured = Pages | Unpictured
