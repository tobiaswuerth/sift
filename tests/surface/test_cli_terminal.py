import io
from typing import override

from sift.interfaces.cli.main import run
from sift.settings import Settings


class TerminalWithoutDescriptor(io.StringIO):
    @override
    def isatty(self) -> bool:
        return True


def test_a_terminal_without_a_descriptor_gets_no_colour_and_runs(settings: Settings) -> None:
    out, err = io.StringIO(), TerminalWithoutDescriptor()
    code = run(("status", "--color", "auto"), settings, out, err)
    assert (code, "\x1b[" in err.getvalue(), "fileno" in err.getvalue()) == (0, False, False)
