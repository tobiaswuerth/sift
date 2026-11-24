import os
import threading
from collections.abc import Callable
from typing import NoReturn, final, override

from sift.domain.text.documents import Outcome, Refused
from sift.ingest.readers import ToRead

CRASHED_EXIT = 3


def echoed(asked: ToRead) -> Outcome:
    return Refused(asked.path.name, read_again=False)


def misbehaving(asked: ToRead) -> Outcome:
    match asked.path.stem:
        case "crash":
            os._exit(CRASHED_EXIT)
        case "hang":
            _ = threading.Event().wait()
        case "raise":
            raise ValueError("a reader's bug")
        case _:
            pass
    return echoed(asked)


def echoed_after(_unpickled: object, asked: ToRead) -> Outcome:
    return echoed(asked)


@final
class EndsWhenUnpickled:
    @override
    def __reduce__(self) -> tuple[Callable[[int], NoReturn], tuple[int]]:
        return os._exit, (CRASHED_EXIT,)


@final
class StuckWhenUnpickled:
    def __init__(self, wait: Callable[[], bool]) -> None:
        self.wait = wait

    @override
    def __reduce__(self) -> tuple[Callable[[], bool], tuple[()]]:
        return self.wait, ()
