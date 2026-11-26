import subprocess
import sys
from pathlib import Path

import pytest

from tests.environments import clean_environment
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


READ_FROM_ELSEWHERE = """
from tools import check_idioms, check_prose, fts5_table
from tools.findings import PYTHON_ROOTS, files

assert len(files(PYTHON_ROOTS, (".py",))) > 100
assert "UNREADABLE" in check_idioms.first_party_tuples("sift.ingest.mail")
assert check_prose.main([]) == 0
assert fts5_table.MODULE.is_file()
"""


def test_the_checks_read_the_repository_from_any_folder(tmp_path: Path) -> None:
    environment = clean_environment(PYTHONPATH=str(ROOT))
    for argv in ([sys.executable, "-c", READ_FROM_ELSEWHERE], [sys.executable, "-m", "tools", "limits"]):
        ran = subprocess.run(
            argv, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120, check=False
        )
        assert ran.returncode == 0, (argv, ran.stdout, ran.stderr)
