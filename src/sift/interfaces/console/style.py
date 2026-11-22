import codecs
from dataclasses import dataclass
from enum import Enum, StrEnum

GLYPHS = "\N{BOX DRAWINGS HEAVY HORIZONTAL}\N{BOX DRAWINGS LIGHT HORIZONTAL}\N{MIDDLE DOT}"
RESET = "\x1b[0m"


class Tone(Enum):
    STRONG = "1"
    SECONDARY = "2"
    ERROR = "31"
    DONE = "32"
    WARNING = "33"


class Colour(StrEnum):
    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"


@dataclass(frozen=True, slots=True)
class Style:
    colour: bool = False
    unicode: bool = True


PLAIN = Style()


def paint(text: str, tone: Tone, style: Style) -> str:
    return f"\x1b[{tone.value}m{text}{RESET}" if style.colour and text else text


def _prints_glyphs(encoding: str | None) -> bool:
    if encoding is None:
        return True
    try:
        GLYPHS.encode(codecs.lookup(encoding).name)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


@dataclass(frozen=True, slots=True)
class Terminal:
    terminal: bool
    escapes: bool
    encoding: str | None


@dataclass(frozen=True, slots=True)
class Asked:
    choice: Colour
    no_color: bool
    dumb: bool


def style_of(stream: Terminal, asked: Asked) -> Style:
    automatic = stream.terminal and stream.escapes and not asked.no_color and not asked.dumb
    colour = asked.choice is Colour.ALWAYS or (asked.choice is Colour.AUTO and automatic)
    return Style(colour=colour, unicode=_prints_glyphs(stream.encoding))
