import ctypes
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from sift.interfaces.console.windows import VIRTUAL_TERMINAL_PROCESSING, Console, console, escapes_work, prepared


@dataclass(slots=True)
class Fake:
    answer: object = 1
    mode: int | None = None
    argtypes: object = None
    restype: object = None
    calls: list[tuple[object, ...]] = field(default_factory=list[tuple[object, ...]])

    def __call__(self, *arguments: object) -> object:
        self.calls.append(arguments)
        written = arguments[-1]
        if self.mode is not None and isinstance(written, ctypes.c_ulong):
            written.value = self.mode
        return self.answer


def faked(mode: int | None, *, readable: bool = True, settable: bool = True) -> tuple[Console, Fake]:
    setting = Fake(answer=int(settable))
    return Console(Fake(answer=77), Fake(answer=int(readable), mode=mode), setting), setting


def test_off_windows_escapes_work_as_they_are() -> None:
    assert escapes_work(None, 2)


def test_a_console_that_processes_escapes_is_left_as_it_is() -> None:
    found, setting = faked(VIRTUAL_TERMINAL_PROCESSING | 3)
    assert escapes_work(found, 2)
    assert setting.calls == []


def test_a_console_that_does_not_is_asked_to() -> None:
    found, setting = faked(3)
    assert escapes_work(found, 1)
    assert setting.calls == [(77, 3 | VIRTUAL_TERMINAL_PROCESSING)]
    refused, _ = faked(3, settable=False)
    assert not escapes_work(refused, 1)


@pytest.mark.parametrize("descriptor, readable", [(2, False), (5, True)])
def test_a_stream_that_is_no_console_takes_no_escapes(descriptor: int, *, readable: bool) -> None:
    found, _ = faked(0, readable=readable)
    assert not escapes_work(found, descriptor)


def test_the_console_functions_are_declared_with_their_types() -> None:
    library = SimpleNamespace(GetStdHandle=Fake(), GetConsoleMode=Fake(), SetConsoleMode=Fake())
    found = prepared(library)
    assert found.standard_handle.restype is ctypes.c_void_p
    assert found.mode.argtypes == (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
    assert found.set_mode.argtypes == (ctypes.c_void_p, ctypes.c_ulong)


def test_this_system_s_console_is_reached_on_windows_alone() -> None:
    found = console()
    assert (found is None) == (sys.platform != "win32")
    assert escapes_work(found, 5) is (found is None)
