from importlib import import_module
from typing import Protocol, cast

from sift.domain.text.documents import Conversion


class Converter(Protocol):
    def converted(self, raw: bytes, name: str) -> Conversion: ...


def _as_converter(module: object) -> Converter:
    return cast("Converter", module)


def convert(raw: bytes, name: str) -> Conversion:
    return _as_converter(import_module("sift.ingest.converter")).converted(raw, name)
