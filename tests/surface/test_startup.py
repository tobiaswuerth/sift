import os
import subprocess
import sys
from pathlib import Path

from sift.interfaces.cli.main import ENTRIES, command_of
from sift.settings import Settings
from tests.surface.commands import indexed_notes

LOADED = (
    "import sys; before = set(sys.modules); from sift.interfaces.cli.main import run; "
    "from sift.settings import from_environment; run(sys.argv[1:], from_environment(), None, None); "
    "print(sorted({name.split('.')[0] for name in set(sys.modules) - before} - set(sys.stdlib_module_names)))"
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
