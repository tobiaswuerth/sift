import threading
from collections.abc import Callable
from typing import Protocol

STEP = 0.01
WAITED = 30.0


class Joinable(Protocol):
    def join(self, timeout: float | None = None) -> None: ...

    def is_alive(self) -> bool: ...


class Awaitable(Protocol):
    def wait(self, timeout: float | None = None) -> bool: ...


def paused(seconds: float) -> None:
    threading.Event().wait(seconds)


def until(holds: Callable[[], bool], seconds: float = WAITED) -> bool:
    for _ in range(round(seconds / STEP)):
        if holds():
            return True
        paused(STEP)
    return holds()


def joined(each: Joinable, seconds: float = WAITED) -> None:
    each.join(seconds)
    assert not each.is_alive(), f"{each!r} still runs after {seconds} s"


def still_running(each: Joinable, seconds: float) -> None:
    each.join(seconds)
    assert each.is_alive(), f"{each!r} ended within {seconds} s, though it should still wait"


def waited(event: Awaitable, seconds: float = WAITED) -> None:
    assert event.wait(seconds), f"{event!r} was not set within {seconds} s"


def crossed(barrier: threading.Barrier, seconds: float = WAITED) -> None:
    barrier.wait(seconds)
