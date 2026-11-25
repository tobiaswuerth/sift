import os
import subprocess
import sys
from pathlib import Path

import pytest

from sift.api.errors import InvalidRequestError
from sift.interfaces.cli.entries import ENTRIES, command_of
from sift.interfaces.cli.main import main
from sift.settings import OCR_REMEDY, Settings
from tests.surface.commands import indexed_notes, ran

MAIN_AGAIN = "__mp_main__"
UNLISTED_BY_PYTHON_3_12_0 = "_wmi"
LOADED = (
    "import sys; before = set(sys.modules); from sift.interfaces.cli.main import run; "
    "from sift.settings import from_environment; run(sys.argv[1:], from_environment(), None, None); "
    "print(sorted({name.split('.')[0] for name in set(sys.modules) - before} "
    f"- set(sys.stdlib_module_names) - {{{MAIN_AGAIN!r}, {UNLISTED_BY_PYTHON_3_12_0!r}}}))"
)


def loaded(home: Path, *argv: str) -> str:
    environment = {**os.environ, "SIFT_HOME": str(home)}
    found = subprocess.run(
        [sys.executable, "-c", LOADED, *argv], capture_output=True, text=True, check=True, timeout=60, env=environment
    )
    return found.stdout


def test_help_is_plain_where_python_would_colour_it(settings: Settings) -> None:
    coloured = {name: value for name, value in os.environ.items() if name != "NO_COLOR"}
    environment = {**coloured, "FORCE_COLOR": "1", "SIFT_HOME": str(settings.home), "PYTHONIOENCODING": "utf-8"}
    for argv in (("--help",), ("search", "--help"), ()):
        found = subprocess.run(
            [sys.executable, "-m", "sift", *argv],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=60,
            env=environment,
        )
        assert found.stdout.startswith("usage: sift"), argv


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


def _no_ocr_known() -> Settings:
    raise InvalidRequestError("SIFT_OCR is 'maybe'", OCR_REMEDY, "SIFT_OCR")


@pytest.mark.parametrize("argv", [["--help"], ["--version"], ["search", "--help"], []])
def test_help_and_the_version_are_shown_whatever_the_settings_say(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as ended:
        main(argv, _no_ocr_known)
    assert (ended.value.code, capsys.readouterr().out.startswith(("usage: sift", "sift "))) == (0, True)


def test_settings_refused_are_said_once_a_command_is_known(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ended:
        main(["status"], _no_ocr_known)
    assert (ended.value.code, capsys.readouterr().err.startswith("error  SIFT_OCR is 'maybe'")) == (2, True)
