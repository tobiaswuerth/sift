import ctypes
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable

VIRTUAL_TERMINAL_PROCESSING = 0x0004
STANDARD_HANDLES = MappingProxyType({1: -11, 2: -12})


class Function(Protocol):
    argtypes: object
    restype: object

    def __call__(self, *arguments: object) -> object: ...


@dataclass(frozen=True, slots=True)
class Console:
    standard_handle: Function
    mode: Function
    set_mode: Function


def _function(library: object, name: str, argtypes: tuple[object, ...], restype: object) -> Function:
    found = cast("Function", getattr(library, name))
    found.argtypes = argtypes
    found.restype = restype
    return found


def prepared(library: object) -> Console:
    return Console(
        _function(library, "GetStdHandle", (ctypes.c_long,), ctypes.c_void_p),
        _function(library, "GetConsoleMode", (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)), ctypes.c_int),
        _function(library, "SetConsoleMode", (ctypes.c_void_p, ctypes.c_ulong), ctypes.c_int),
    )


def console() -> Console | None:
    load = cast("Callable[[str], object] | None", getattr(ctypes, "WinDLL", None))
    return None if load is None else prepared(load("kernel32"))


def escapes_work(found: Console | None, descriptor: int) -> bool:
    if found is None:
        return True
    which = STANDARD_HANDLES.get(descriptor)
    if which is None:
        return False
    handle, mode = found.standard_handle(which), ctypes.c_ulong()
    if not found.mode(handle, mode):
        return False
    if mode.value & VIRTUAL_TERMINAL_PROCESSING:
        return True
    return bool(found.set_mode(handle, mode.value | VIRTUAL_TERMINAL_PROCESSING))
