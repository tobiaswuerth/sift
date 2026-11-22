from collections.abc import Iterator
from types import MappingProxyType

from sift.domain.text.fts5_table import FOLDS, SEPARATORS


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


def tokens(text: str) -> list[str]:
    return text.translate(TRANSLATION).split()
