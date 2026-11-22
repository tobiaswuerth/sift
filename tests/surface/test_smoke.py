import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast


def sift(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sift", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
        env={**os.environ, "SIFT_HOME": str(home), "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"},
    )


def test_every_command_runs_as_a_process_from_a_folder_to_its_documents(tmp_path: Path) -> None:
    home, notes = tmp_path / "home", tmp_path / "Notizen Müller"
    (notes / "sub").mkdir(parents=True)
    (notes / "rl.md").write_bytes(b"# Reward shaping\n\nSpeeds up learning.\n")
    (notes / "sub" / "brush.md").write_bytes(b"# Brush\n")
    (notes / "huge.txt").write_bytes(b"x")

    added = sift(home, "add", str(notes), "--name", "notes")
    assert (added.returncode, added.stdout) == (0, "")
    assert added.stderr.split("\n")[0] == f"added  notes  {notes.resolve()}"

    indexed = sift(home, "index")
    assert (indexed.returncode, indexed.stdout) == (0, "")
    assert "  done  +3  ~0  -0  " in indexed.stderr

    assert sift(home, "index", "notes", "--json").stdout.count('"added": 0') == 1

    status = sift(home, "status")
    assert (status.returncode, status.stderr) == (0, "")
    assert status.stdout.split("\n")[1].split()[-4:-2] == ["3", "3"]
    listed = sift(home, "status", "notes", "--unreadable", "--json")
    assert '"unreadable_files": []' in listed.stdout
    assert json.loads(listed.stdout)

    one = sift(home, "get", "notes/brush.md")
    assert (one.returncode, one.stdout, one.stderr) == (0, "# Brush\n", "")
    lines = sift(home, "get", str(notes / "rl.md"), "notes/sub/brush.md", "--lines", "1")
    assert lines.stdout == "==> notes/rl.md <==\n# Reward shaping\n\n==> notes/sub/brush.md <==\n# Brush\n"

    found = sift(home, "search", "learning", "-k", "1")
    assert (found.returncode, found.stdout.split("\n")[1]) == (0, " " * 9 + "notes/rl.md:1-3")
    assert found.stderr.startswith("1 result  \N{MIDDLE DOT}  keyword  \N{MIDDLE DOT}  1 collection")
    assert json.loads(sift(home, "search", '"speeds up"', "--json").stdout)["hits"][0]["id"] == "notes/rl.md"
    timed = cast("dict[str, object]", json.loads(sift(home, "search", "speed*", "--budget", "5", "--json").stdout))
    assert timed["partial"] is False
    assert cast("list[dict[str, object]]", timed["hits"])[0]["id"] == "notes/rl.md"

    missing = sift(home, "get", "notes/gone.md")
    assert (missing.returncode, missing.stdout) == (1, "")
    assert missing.stderr.startswith("error  notes/gone.md: unknown document")
    assert sift(home, "frobnicate").returncode == 2
    assert sift(home, "--help").stdout.startswith("usage: sift")
