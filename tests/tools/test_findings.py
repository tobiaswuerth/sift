import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.tools.repositories import tracked_in
from tools.findings import ROOT, files, read


def test_files_are_those_git_tracks(tmp_path: Path) -> None:
    (tmp_path / "tracked.py").write_bytes(b"x = 1\n")
    (tmp_path / "untracked.py").write_bytes(b"y = 2\n")
    (tmp_path / "notes.md").write_bytes(b"# notes\n")
    tracked_in(tmp_path, "tracked.py", "notes.md")
    assert files((".",), (".py",), tmp_path) == ["tracked.py"]
    assert read("tracked.py", tmp_path) == "x = 1\n"


def test_roots_with_no_file_tracked_are_an_error(tmp_path: Path) -> None:
    tracked_in(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"git tracks no \.py file under src"):
        files(("src",), (".py",), tmp_path)


def test_the_checks_read_the_repository_from_any_folder(tmp_path: Path) -> None:
    counted = "from tools.findings import PYTHON_ROOTS, files; print(len(files(PYTHON_ROOTS, ('.py',))))"
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    for argv in ([sys.executable, "-c", counted], [sys.executable, "-m", "tools", "twins"]):
        ran = subprocess.run(
            argv, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120, check=False
        )
        assert ran.returncode == 0, ran.stderr
    listed = subprocess.run(
        [sys.executable, "-c", counted],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert int(listed.stdout) > 100
