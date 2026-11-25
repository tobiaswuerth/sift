import re
from dataclasses import dataclass
from enum import Enum, StrEnum

RESET = "\x1b[0m"
PAINTED = re.compile("\x1b\\[[0-9;]*m")


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


PLAIN = Style()


def paint(text: str, tone: Tone, style: Style) -> str:
    return f"\x1b[{tone.value}m{text}{RESET}" if style.colour and text else text


def unpainted(text: str) -> str:
    return PAINTED.sub("", text)


@dataclass(frozen=True, slots=True)
class Terminal:
    terminal: bool
    escapes: bool


@dataclass(frozen=True, slots=True)
class Asked:
    choice: Colour
    no_color: bool
    dumb: bool


def style_of(stream: Terminal, asked: Asked) -> Style:
    automatic = stream.terminal and stream.escapes and not asked.no_color and not asked.dumb
    colour = asked.choice is Colour.ALWAYS or (asked.choice is Colour.AUTO and automatic)
    return Style(colour=colour)
