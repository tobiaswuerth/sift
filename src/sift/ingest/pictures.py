from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast


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


class Painter(Protocol):
    def pages(self, raw: bytes, most: int) -> Pictured: ...


def _as_painter(module: object) -> Painter:
    return cast("Painter", module)


def picture(asked: ToPicture) -> Pictured:
    return _as_painter(import_module("sift.ingest.pages")).pages(asked.raw, asked.most)
