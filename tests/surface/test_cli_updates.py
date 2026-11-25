import dataclasses
import io
from importlib.metadata import version
from pathlib import Path

from sift.interfaces.cli.main import Host, run
from sift.interfaces.cli.updates import PYPI, Checking
from sift.settings import Settings
from tests.surface.commands import Terminal, ran

NOW = 1_800_000_000.0


def test_in_a_terminal_a_newer_sift_is_told_of_once_a_day_after_what_the_command_said_when_asked_for(
    settings: Settings, notes: Path, tmp_path: Path
) -> None:
    on = dataclasses.replace(settings, update_check=True)
    assert ran(on, "add", str(notes))[0] == 0
    asked = list[tuple[str, str]]()

    def fetch(url: str, installed: str) -> str:
        asked.append((url, installed))
        return "99.0.0"

    checking = Checking(
        now=lambda: NOW,
        fetch=fetch,
        prefix=tmp_path / "venv",
        package=tmp_path / "venv" / "sift",
        pip=lambda: True,
        python="python",
    )
    host = Host(checking=checking)
    with Terminal((tmp_path / "err").open("w+b"), encoding="utf-8", newline="\n") as err:
        assert run(("status", "--json"), on, io.StringIO(), err, host) == 0
        assert run(("status",), settings, io.StringIO(), err, host) == 0
        assert asked == []
        for argv in (("rm", "nope"), ("status",)):
            run((*argv, "--color", "never"), on, io.StringIO(), err, host)
        err.flush()
        said = (tmp_path / "err").read_bytes().decode().split("\n")
    told = f" note  sift 99.0.0 is out, this is {version('sift-rag')}: python -m pip install --upgrade sift-rag"
    assert (said[0][:7], said[1][:7], said[2:]) == ("error  ", "  fix  ", [told, ""])
    assert asked == [(PYPI, version("sift-rag"))]


def test_results_are_written_before_pypi_is_asked(settings: Settings, notes: Path, tmp_path: Path) -> None:
    on = dataclasses.replace(settings, update_check=True)
    assert ran(on, "add", str(notes))[0] == 0
    out = io.StringIO()
    shown_first = list[bool]()

    def fetch(_url: str, _installed: str) -> str:
        shown_first.append("notes" in out.getvalue())
        return "0.0.1"

    checking = Checking(now=lambda: NOW, fetch=fetch, prefix=tmp_path / "venv", package=tmp_path / "venv" / "sift")
    with Terminal((tmp_path / "err").open("w+b"), encoding="utf-8", newline="\n") as err:
        assert run(("status",), on, out, err, Host(checking=checking)) == 0
    assert shown_first == [True]
