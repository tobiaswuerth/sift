import shlex
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Protocol

from sift.late import late

WINDOWS = "win32"
PROCESSES = "subprocess"


class CommandLines(Protocol):
    def list2cmdline(self, seq: Iterable[str]) -> str: ...


def command(words: Sequence[str], platform: str) -> str:
    if platform != WINDOWS:
        return shlex.join(words)
    return late(PROCESSES, CommandLines).list2cmdline(words)


if TYPE_CHECKING:
    import subprocess as _subprocess

    _PROCESSES: CommandLines = _subprocess
