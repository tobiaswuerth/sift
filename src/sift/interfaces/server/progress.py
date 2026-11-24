import math
from collections.abc import Callable
from dataclasses import dataclass

from sift.api.events import Advanced, Event, Reporter

PROGRESS_EVERY_SECONDS = 0.1


@dataclass(slots=True, eq=False)
class Coalesced:
    report: Reporter
    clock: Callable[[], float]
    every: float = PROGRESS_EVERY_SECONDS
    last: float = -math.inf
    held: Advanced | None = None

    def __call__(self, event: Event) -> None:
        if isinstance(event, Advanced):
            now = self.clock()
            if now - self.last < self.every:
                self.held = event
                return
            self.last, self.held = now, None
        else:
            self.flush()
        self.report(event)

    def flush(self) -> None:
        if self.held is not None:
            held, self.held = self.held, None
            self.report(held)
