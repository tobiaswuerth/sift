from typing import TYPE_CHECKING, Protocol

from sift.domain.text.documents import Conversion
from sift.late import late

CONVERTER = "sift.ingest.converter"


class Converter(Protocol):
    def converted(self, raw: bytes, name: str) -> Conversion: ...


def convert(raw: bytes, name: str) -> Conversion:
    return late(CONVERTER, Converter).converted(raw, name)


if TYPE_CHECKING:
    from sift.ingest import converter as _converter

    _CONVERTER: Converter = _converter
