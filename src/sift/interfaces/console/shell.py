import shlex
from collections.abc import Iterable, Sequence
from importlib import import_module
from typing import Protocol, cast

WINDOWS = "win32"
PROCESSES = "subprocess"


class CommandLines(Protocol):
    def list2cmdline(self, seq: Iterable[str]) -> str: ...


def _as_command_lines(module: object) -> CommandLines:
    return cast("CommandLines", module)


def command(words: Sequence[str], platform: str) -> str:
    if platform != WINDOWS:
        return shlex.join(words)
    return _as_command_lines(import_module(PROCESSES)).list2cmdline(words)
