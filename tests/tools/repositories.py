import shutil
import subprocess
from pathlib import Path


def _git() -> str:
    found = shutil.which("git")
    if found is None:
        raise FileNotFoundError("git is not on the PATH: install it to run the tests of the checks")
    return found


def tracked_in(folder: Path, *names: str) -> Path:
    subprocess.run([_git(), "init", "-q"], cwd=folder, check=True, capture_output=True, timeout=60)
    if names:
        subprocess.run([_git(), "add", "--", *names], cwd=folder, check=True, capture_output=True, timeout=60)
    return folder
