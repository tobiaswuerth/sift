import os
import subprocess
import sys
from pathlib import Path

import pytest

from sift.api.errors import InvalidRequestError
from sift.interfaces.cli.main import ENTRIES, command_of, main
from sift.settings import Settings
from tests.surface.commands import indexed_notes, ran

MAIN_AGAIN = "__mp_main__"
LOADED = (
    "import sys; before = set(sys.modules); from sift.interfaces.cli.main import run; "
    "from sift.settings import from_environment; run(sys.argv[1:], from_environment(), None, None); "
    "print(sorted({name.split('.')[0] for name in set(sys.modules) - before} "
    f"- set(sys.stdlib_module_names) - {{{MAIN_AGAIN!r}}}))"
)


def loaded(home: Path, *argv: str) -> str:
    environment = {**os.environ, "SIFT_HOME": str(home)}
    found = subprocess.run(
        [sys.executable, "-c", LOADED, *argv], capture_output=True, text=True, check=True, timeout=60, env=environment
    )
    return found.stdout


def test_a_command_loads_what_it_needs_alone(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    for argv in (
        ("--help",),
        ("--version",),
        ("status",),
        ("get", "notes/rl.md"),
        ("add", str(notes)),
        ("links", "rl"),
    ):
        assert loaded(settings.home, *argv) == "['sift']\n", argv
    for argv in (("search", "brush"), ("index",), ("mentions", "notes/rl.md")):
        assert loaded(settings.home, *argv) == "['numpy', 'sift']\n", argv


def test_every_command_of_the_table_is_made_where_the_table_says() -> None:
    assert all(callable(command_of(name).run) for name in ENTRIES)


def _no_home() -> Settings:
    raise InvalidRequestError("no home folder is set", "set SIFT_HOME to an absolute folder")


def test_settings_that_cannot_be_read_end_sift_saying_how_to_set_them(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ended:
        main(["status"], _no_home)
    said = capsys.readouterr().err
    assert (ended.value.code, said) == (2, "error  no home folder is set\n  fix  set SIFT_HOME to an absolute folder\n")


def test_a_path_from_the_user_s_folder_needs_one(tmp_path: Path) -> None:
    settings = Settings(home=tmp_path / "home", user=None, cwd=tmp_path)
    code, _, err = ran(settings, "add", "~/notes")
    assert (code, "no home folder is set: set SIFT_HOME to an absolute folder" in err) == (2, True)
