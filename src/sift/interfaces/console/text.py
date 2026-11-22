import unicodedata
from collections.abc import Sequence
from types import MappingProxyType

GAP = "  "
LABEL_COLUMNS = 5
WIDE = frozenset({"W", "F"})
NO_WIDTH = frozenset({"Mn", "Me", "Cf"})
ESCAPES = MappingProxyType({"\n": "\\n", "\r": "\\r", "\t": "\\t"})


def _columns(character: str) -> int:
    if unicodedata.category(character) in NO_WIDTH:
        return 0
    return 2 if unicodedata.east_asian_width(character) in WIDE else 1


def width(text: str) -> int:
    return sum(map(_columns, text))


def _escaped(character: str) -> str:
    if unicodedata.category(character) != "Cc":
        return character
    return ESCAPES.get(character) or f"\\x{ord(character):02x}"


def printable(text: str) -> str:
    return "".join(map(_escaped, text))


def _padded(text: str, columns: int, *, right: bool) -> str:
    padding = " " * (columns - width(text))
    return padding + text if right else text + padding


def table(rows: Sequence[Sequence[str]], right: frozenset[int]) -> tuple[str, ...]:
    shown = [[printable(cell) for cell in row] for row in rows]
    columns = [max(map(width, column)) for column in zip(*shown, strict=True)]
    return tuple(
        GAP.join(_padded(cell, columns[index], right=index in right) for index, cell in enumerate(row)) for row in shown
    )


def labelled(label: str, text: str) -> str:
    return f"{label:>{LABEL_COLUMNS}}{GAP}{printable(text)}"
