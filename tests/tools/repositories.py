import shutil
import subprocess
from pathlib import Path


def git_path() -> str:
    found = shutil.which("git")
    if found is None:
        raise FileNotFoundError("git is not on the PATH: install it to run the tests of the checks")
    return found


def tracked_in(folder: Path, *names: str) -> Path:
    subprocess.run([git_path(), "init", "-q"], cwd=folder, check=True, capture_output=True, timeout=60)
    if names:
        subprocess.run([git_path(), "add", "--", *names], cwd=folder, check=True, capture_output=True, timeout=60)
    return folder
