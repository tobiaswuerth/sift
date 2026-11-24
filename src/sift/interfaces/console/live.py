import threading
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from sift.api.events import Advanced, Event
from sift.interfaces.console.progress import Looks, View, running_line, shown
from sift.interfaces.console.streams import Sink, draw, say
from sift.interfaces.console.text import width

REDRAW_SECONDS = 0.1
HEARTBEAT_SECONDS = 60.0
TICK_SECONDS = 0.5
STOP_SECONDS = 5.0


@dataclass(slots=True)
class Live:
    stream: Sink | None
    looks: Looks
    redraw: bool
    columns: Callable[[], int]
    clock: Callable[[], float]
    view: View = field(default_factory=View)
    drawn: int = 0
    last_drawn: float = float("-inf")
    last_said: float = float("-inf")
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, event: Event) -> None:
        with self.lock:
            now = self.clock()
            self.view, lines = shown(self.view, event, now, self.looks)
            if lines:
                self._said(lines, now)
            if self.redraw and (not isinstance(event, Advanced) or now - self.last_drawn >= REDRAW_SECONDS):
                self._drawn(now)

    def tick(self) -> None:
        with self.lock:
            running = self.view.running
            if running is None:
                return
            now = self.clock()
            if self.redraw:
                self._drawn(now)
                return
            line = running_line(self.view, now, self.columns(), self.looks.style)
            if line is not None and now - max(self.last_said, running.started) >= HEARTBEAT_SECONDS:
                self._said((line.rstrip(),), now)

    def close(self) -> None:
        with self.lock:
            self._erased()

    def _said(self, lines: Sequence[str], now: float) -> None:
        self._erased()
        say(self.stream, lines)
        self.last_said = now

    def _drawn(self, now: float) -> None:
        line = running_line(self.view, now, self.columns(), self.looks.style)
        if line is None:
            self._erased()
            return
        draw(self.stream, "\r" + line + " " * max(self.drawn - width(line), 0))
        self.drawn, self.last_drawn = width(line), now

    def _erased(self) -> None:
        if self.drawn:
            draw(self.stream, "\r" + " " * self.drawn + "\r")
            self.drawn = 0


@contextmanager
def ticking(tick: Callable[[], None], interval: float = TICK_SECONDS) -> Generator[None]:
    stop = threading.Event()

    def run() -> None:
        while not stop.wait(interval):
            tick()

    thread = threading.Thread(target=run, name="sift-ticker", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(STOP_SECONDS)
