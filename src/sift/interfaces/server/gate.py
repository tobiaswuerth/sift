import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

from sift.api.errors import Problem, StoreUnavailableError
from sift.api.events import Noted, Reporter

ALONE_PATIENCE_SECONDS = 300.0
LOOK_SECONDS = 0.2
WAITING_FOR_READS = "searches read this home's stores; this command waits for them to finish"
WAITING_FOR_ALONE = "another command removes or moves a store of this home; this one waits for it to finish"
STILL_READ = "searches still read this home's stores after {seconds:g} s"
STILL_READ_REMEDY = "ask again once they are done"


def _always() -> None:
    return None


@dataclass(slots=True, eq=False)
class Gate:
    condition: threading.Condition = field(default_factory=threading.Condition)
    readers: int = 0
    held_alone: bool = False

    @contextmanager
    def shared(self, report: Reporter) -> Generator[None]:
        with self.condition:
            if self.held_alone:
                report(Noted(WAITING_FOR_ALONE))
            self.condition.wait_for(lambda: not self.held_alone)
            self.readers += 1
        try:
            yield
        finally:
            with self.condition:
                self.readers -= 1
                self.condition.notify_all()

    @contextmanager
    def alone(
        self, report: Reporter, patience: float = ALONE_PATIENCE_SECONDS, alive: Callable[[], None] = _always
    ) -> Generator[None]:
        with self.condition:
            self.condition.wait_for(lambda: not self.held_alone)
            self.held_alone = True
        try:
            with self.condition:
                if self.readers:
                    report(Noted(WAITING_FOR_READS))
                looks = max(1, round(patience / LOOK_SECONDS))
                while not self.condition.wait_for(lambda: self.readers == 0, timeout=patience / looks):
                    alive()
                    looks -= 1
                    if not looks:
                        said = STILL_READ.format(seconds=patience)
                        raise StoreUnavailableError(said, STILL_READ_REMEDY, code=Problem.BUSY)
            yield
        finally:
            with self.condition:
                self.held_alone = False
                self.condition.notify_all()
