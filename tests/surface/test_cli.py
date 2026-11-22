import subprocess
import sys
from importlib.metadata import version

import pytest

from sift.interfaces.cli.main import main


def sift(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sift", *args], capture_output=True, text=True, check=False, timeout=60
    )


def test_version_names_the_installed_release() -> None:
    result = sift("--version")
    assert (result.returncode, result.stdout, result.stderr) == (0, f"sift {version('sift-rag')}\n", "")


def test_without_a_command_the_usage_is_shown(capsys: pytest.CaptureFixture[str]) -> None:
    main([])
    assert capsys.readouterr().out.startswith("usage: sift")


def test_an_unknown_option_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["--frobnicate"])
    assert stopped.value.code == 2
    assert "unrecognized arguments: --frobnicate" in capsys.readouterr().err
