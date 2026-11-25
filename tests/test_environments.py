import os
import subprocess
import sys
from pathlib import Path

from tests.environments import clean_environment

SHELL = {"SIFT_OCR": "sometimes", "CUDA_VISIBLE_DEVICES": "-1", "TERM": "dumb", "SIFT_UPDATE_CHECK": "on"}


def test_a_child_keeps_what_runs_it_and_the_session_s_folders_and_nothing_of_the_shell() -> None:
    shell = {
        **SHELL,
        "PATH": "p",
        "COVERAGE_PROCESS_CONFIG": "c",
        "SIFT_HOME": "h",
        "LOCALAPPDATA": "s",
        "NO_COLOR": "1",
    }
    assert clean_environment(shell, PYTHONIOENCODING="utf-8") == {
        "PATH": "p",
        "COVERAGE_PROCESS_CONFIG": "c",
        "SIFT_HOME": "h",
        "LOCALAPPDATA": "s",
        "SIFT_EMBED_MODEL": "",
        "SIFT_RERANK_MODEL": "",
        "SIFT_OCR": "",
        "SIFT_UPDATE_CHECK": "",
        "PYTHONIOENCODING": "utf-8",
    }


def test_a_child_started_from_a_shell_full_of_settings_sees_sift_s_defaults(tmp_path: Path) -> None:
    shown = (
        "from sift.settings import from_environment as f; s = f(); print(s.ocr.value, s.update_check, s.dumb, s.gpus)"
    )
    environment = clean_environment({**os.environ, **SHELL}, SIFT_HOME=str(tmp_path / "home"))
    ran = subprocess.run(
        [sys.executable, "-c", shown], env=environment, capture_output=True, text=True, check=True, timeout=60
    )
    assert ran.stdout.split() == ["auto", "False", "False", "None"]
