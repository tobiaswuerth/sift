import threading
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(slots=True, eq=False)
class Activity:
    clock: Callable[[], float]
    lock: threading.Lock = field(default_factory=threading.Lock)
    asking: int = 0
    last: float = field(init=False)

    def __post_init__(self) -> None:
        self.last = self.clock()

    def began(self) -> None:
        with self.lock:
            self.asking += 1

    def ended(self) -> None:
        with self.lock:
            self.asking -= 1
            self.last = self.clock()

    def idle_for(self) -> float:
        with self.lock:
            return 0.0 if self.asking else self.clock() - self.last
