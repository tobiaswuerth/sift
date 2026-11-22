import subprocess
import sys

import pytest

from tools.__main__ import CHECKS, main


def tools(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools", *args], capture_output=True, text=True, check=False, timeout=60
    )


def test_a_check_runs_by_its_name_and_exits_with_its_result() -> None:
    result = tools("prose")
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


@pytest.mark.parametrize("argv", [[], ["nothing"]])
def test_without_a_known_check_the_usage_names_them_all(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 2
    shown = capsys.readouterr().out
    assert all(name in shown for name in CHECKS)
