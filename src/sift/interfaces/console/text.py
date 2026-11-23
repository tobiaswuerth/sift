import re
import unicodedata
from collections.abc import Sequence
from types import MappingProxyType

from sift.interfaces.console.style import PLAIN, Style, Tone, paint

GAP = "  "
LABEL_COLUMNS = 5
WIDE = frozenset({"W", "F"})
NO_WIDTH = frozenset({"Mn", "Me", "Cf"})
ESCAPES = MappingProxyType({"\n": "\\n", "\r": "\\r", "\t": "\\t"})
FORMATS = (
    "\u00ad\u0600-\u0605\u061c\u06dd\u070f\u0890-\u0891\u08e2\u180e\u200b-\u200c\u200e-\u200f\u202a-"
    "\u202e\u2060-\u2064\u2066-\u206f\ufeff\ufff9-\ufffb\U000110bd\U000110cd\U00013430-\U0001343f"
    "\U0001bca0-\U0001bca3\U0001d173-\U0001d17a\U000e0001\U000e0020-\U000e007f\u2028\u2029\ud800-"
    "\udfff"
)
STEERING = re.compile(f"[\x00-\x1f\x7f-\x9f{FORMATS}]")
STEERING_IN_A_TEXT = re.compile(f"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f{FORMATS}]|\r(?!\n)")
STEERING_IN_JSON = re.compile(f"[\x7f-\x9f{FORMATS}]")
JOINER = "\N{ZERO WIDTH JOINER}"
PRESENTED = "\N{VARIATION SELECTOR-16}"
MODIFIERS = ("\N{EMOJI MODIFIER FITZPATRICK TYPE-1-2}", "\N{EMOJI MODIFIER FITZPATRICK TYPE-6}")
JAMO_AFTER_THE_FIRST = ("\N{HANGUL JUNGSEONG FILLER}", "\u11ff")
UTF16_LOW = 0x10000
LATIN_1_PRINTABLE = 0xA0
NARROW, WIDE_COLUMNS = 1, 2


def _columns(character: str) -> int:
    if unicodedata.category(character) in NO_WIDTH or JAMO_AFTER_THE_FIRST[0] <= character <= JAMO_AFTER_THE_FIRST[1]:
        return 0
    return WIDE_COLUMNS if unicodedata.east_asian_width(character) in WIDE else NARROW


def width(text: str) -> int:
    total, last, joined = 0, 0, False
    for character in text:
        if joined:
            joined = False
        elif character == JOINER:
            joined = last > 0
        elif character == PRESENTED:
            total, last = total + (last == NARROW), max(last, WIDE_COLUMNS * (last > 0))
        elif MODIFIERS[0] <= character <= MODIFIERS[1] and last == WIDE_COLUMNS:
            continue
        else:
            last = _columns(character)
            total += last
    return total


def _escaped(found: re.Match[str]) -> str:
    character = found.group()
    code = ord(character)
    if code < LATIN_1_PRINTABLE:
        return ESCAPES.get(character) or f"\\x{code:02x}"
    return f"\\u{code:04x}" if code < UTF16_LOW else f"\\U{code:08x}"


def _json_escaped(found: re.Match[str]) -> str:
    code = ord(found.group())
    if code < UTF16_LOW:
        return f"\\u{code:04x}"
    high, low = divmod(code - UTF16_LOW, 0x400)
    return f"\\u{0xD800 + high:04x}\\u{0xDC00 + low:04x}"


def printable(text: str) -> str:
    return STEERING.sub(_escaped, text)


def printable_text(text: str) -> str:
    return STEERING_IN_A_TEXT.sub(_escaped, text)


def printable_json(text: str) -> str:
    return STEERING_IN_JSON.sub(_json_escaped, text)


def _padded(text: str, columns: int, *, right: bool) -> str:
    padding = " " * (columns - width(text))
    return padding + text if right else text + padding


def table(rows: Sequence[Sequence[str]], right: frozenset[int]) -> tuple[str, ...]:
    shown = [[printable(cell) for cell in row] for row in rows]
    columns = [max(map(width, column)) for column in zip(*shown, strict=True)]
    last = len(columns) - 1
    if last not in right:
        columns[last:] = [0]
    return tuple(
        GAP.join(_padded(cell, columns[index], right=index in right) for index, cell in enumerate(row)) for row in shown
    )


def labelled(label: str, text: str, tone: Tone | None = None, style: Style = PLAIN) -> str:
    padding = " " * (LABEL_COLUMNS - len(label))
    return f"{padding}{paint(label, tone, style) if tone else label}{GAP}{printable(text)}"


def fitted(text: str, columns: int) -> str:
    kept: list[str] = []
    used = 0
    for character in text:
        used += _columns(character)
        if used > columns:
            break
        kept.append(character)
    return "".join(kept)
