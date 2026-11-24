import gc
import os
import socket
import struct
import threading
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import NoReturn, final, override

from sift.domain.text.documents import Outcome, Refused
from sift.ingest.files import Identity
from sift.ingest.readers import ToRead

CRASHED_EXIT = 3
FAILED_EXIT = 1
LARGE = 10_000


def echoed(asked: ToRead) -> Outcome:
    return Refused(asked.path.name, read_again=False)


def _cut_short() -> NoReturn:
    held: list[object] = gc.get_objects()
    (answers,) = [each for each in held if isinstance(each, Connection) and each.writable]
    _ = os.write(answers.fileno(), struct.pack("!i", 1 << 20) + b"\x80\x05 cut short")
    os._exit(CRASHED_EXIT)


def misbehaving(asked: ToRead) -> Outcome:
    match asked.path.stem:
        case "crash":
            os._exit(CRASHED_EXIT)
        case "exit1":
            os._exit(FAILED_EXIT)
        case "hang":
            _ = threading.Event().wait()
        case "raise":
            raise ValueError("a reader's bug")
        case "cut":
            _cut_short()
        case "large":
            return Refused("x" * LARGE, read_again=False)
        case _:
            pass
    return echoed(asked)


def pid_or_hang(asked: ToRead) -> Outcome:
    if asked.path.stem == "hang":
        _ = threading.Event().wait()
    return Refused(str(os.getpid()), read_again=False)


def held(asked: ToRead) -> Outcome:
    with socket.create_connection(("127.0.0.1", asked.size)) as connection:
        connection.sendall(b"%d\n" % os.getpid())
        _ = threading.Event().wait()
    return echoed(asked)


def echoed_after(_unpickled: object, asked: ToRead) -> Outcome:
    return echoed(asked)


@final
class EndsWhenUnpickled:
    @override
    def __reduce__(self) -> tuple[Callable[[int], NoReturn], tuple[int]]:
        return os._exit, (CRASHED_EXIT,)


@final
class IdentityEndingWhenUnpickled(Identity):
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
