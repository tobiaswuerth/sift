import shlex
import subprocess
from collections.abc import Sequence

WINDOWS = "win32"


def command(words: Sequence[str], platform: str) -> str:
    return subprocess.list2cmdline(words) if platform == WINDOWS else shlex.join(words)
