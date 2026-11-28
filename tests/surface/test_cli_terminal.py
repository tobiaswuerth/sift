import io
from pathlib import Path
from typing import override

from sift.interfaces.cli.main import run
from sift.settings import Settings
from tests.surface.commands import Terminal


class TerminalWithoutDescriptor(io.StringIO):
    @override
    def isatty(self) -> bool:
        return True


def test_a_terminal_without_a_descriptor_gets_no_colour_and_runs(settings: Settings) -> None:
    out, err = io.StringIO(), TerminalWithoutDescriptor()
    code = run(("status", "--color", "auto"), settings, out, err)
    assert (code, "\x1b[" in err.getvalue(), "fileno" in err.getvalue()) == (0, False, False)


def test_what_sift_says_is_utf8_whatever_the_stream_was_set_to_and_draws_its_glyphs(
    settings: Settings, notes: Path
) -> None:
    (notes / "a.md").write_bytes(b"# A\n\nalpha\n")
    assert run(("add", str(notes)), settings, io.StringIO(), io.StringIO()) == 0
    raw = io.BytesIO()
    with Terminal(raw, encoding="cp1252", newline="\n") as err:
        assert run(("index", "--color", "never"), settings, io.StringIO(), err) == 0
        err.flush()
        said = raw.getvalue().decode("utf-8")
    assert ("\N{MIDDLE DOT}" in said, " | " in said) == (True, False)
