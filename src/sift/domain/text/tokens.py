import re
from collections.abc import Iterator
from types import MappingProxyType

from sift.domain.text.fts5_table import FOLDS, SEPARATORS

ASCII_END = 0x80
BYTES = 0x100
OUTSIDE_ASCII = re.compile(r"[^\x00-\x7f]+")


def _points(span: str) -> range:
    low, _, high = span.partition("-")
    return range(int(low, 16), int(high or low, 16) + 1)


def _folds() -> Iterator[tuple[int, str]]:
    for fold in " ".join(FOLDS).split():
        point, _, folded = fold.partition("=")
        yield int(point, 16), chr(int(folded, 16)) if folded else ""


def _translation() -> MappingProxyType[int, str]:
    separators = {point: " " for span in " ".join(SEPARATORS).split() for point in _points(span)}
    return MappingProxyType({0: " ", **separators, **dict(_folds())})


TRANSLATION = _translation()


def _ascii_bytes() -> tuple[bytes, bytes]:
    kept = [TRANSLATION.get(point, chr(point)) for point in range(ASCII_END)]
    table = bytes(ord(each) if each else point for point, each in enumerate(kept)) + bytes(range(ASCII_END, BYTES))
    return table, bytes(point for point, each in enumerate(kept) if not each)


ASCII_TABLE, ASCII_DROPPED = _ascii_bytes()


def _outside_ascii(found: re.Match[str]) -> str:
    return found.group().translate(TRANSLATION)


def tokens(text: str) -> list[str]:
    narrow = text.encode("utf-8", "surrogatepass").translate(ASCII_TABLE, ASCII_DROPPED)
    if text.isascii():
        return narrow.decode("ascii").split()
    return OUTSIDE_ASCII.sub(_outside_ascii, narrow.decode("utf-8", "surrogatepass")).split()
