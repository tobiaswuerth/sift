from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TextIO

from sift.api.events import Advanced, Event
from sift.interfaces.console.progress import Looks, View, running_line, shown
from sift.interfaces.console.streams import draw, say
from sift.interfaces.console.text import width

REDRAW_SECONDS = 0.1


@dataclass(slots=True)
class Live:
    stream: TextIO | None
    looks: Looks
    redraw: bool
    columns: Callable[[], int]
    clock: Callable[[], float]
    view: View = field(default_factory=View)
    drawn: int = 0
    last_drawn: float = float("-inf")

    def __call__(self, event: Event) -> None:
        now = self.clock()
        self.view, lines = shown(self.view, event, now, self.looks)
        if lines:
            self.close()
            say(self.stream, lines)
        if self.redraw and (not isinstance(event, Advanced) or now - self.last_drawn >= REDRAW_SECONDS):
            self._drawn(now)

    def _drawn(self, now: float) -> None:
        line = running_line(self.view, now, self.columns(), self.looks.style)
        if line is None:
            self.close()
            return
        draw(self.stream, "\r" + line + " " * max(self.drawn - width(line), 0))
        self.drawn, self.last_drawn = width(line), now

    def close(self) -> None:
        if self.drawn:
            draw(self.stream, "\r" + " " * self.drawn + "\r")
            self.drawn = 0
