"""The gate: every check a change must pass, in one command, `python scripts/gate.py`. Green means all passed.

The pre-commit hooks (format, lint, the checks of pyproject.toml and the workflows, spelling), the types of all of
sift, the tests in a process per core with the coverage gate and the rules tests/test_package.py keeps, and
the smoke run of every command as a user runs it (scripts/smoke.py). The first check that fails ends it, and so does
one that takes longer than its minutes: a test that hangs holds nobody for good. CI runs the hooks, the type checker,
the tests and the smoke run the same way, the tests and the smoke run on every system and Python.

The types are checked as on each system sift runs on, whichever runs the gate: what only another system has, or
lacks, is found here, not in CI. pyright reads the packages of the Python that runs the gate, which it is told: left
to itself it asks the `python` on the PATH, another one unless the environment is activated.

The tests run once more on the newest Python CI tests on, where a Python warns of what an older one lets pass: in an
environment of its own, .venv-newest beside .venv, which uv makes on the first run and brings up to date on each.
The gate owns that folder; deleting it costs the next run its install."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SYSTEMS = ("Linux", "Darwin", "Windows")  # as pyright names them
# Each check, its command, and the minutes it may take: many times what it takes, the hooks with their first install
CHECKS = (
    ("the pre-commit hooks", [sys.executable, "-m", "pre_commit", "run", "--all-files", "--show-diff-on-failure"], 15),
    *(
        (
            f"the types as on {system}",
            [sys.executable, "-m", "pyright", "--pythonplatform", system, "--pythonpath", sys.executable],
            10,
        )
        for system in SYSTEMS
    ),
    ("the tests, with the coverage gate,", [sys.executable, "-m", "pytest", "-q", "--cov", "-n", "auto"], 20),
    ("the smoke run", [sys.executable, str(Path(__file__).parent / "smoke.py")], 10),
)
# The newest Python CI tests on: the last of its matrix
NEWEST = re.search(r'python: \[.*"([\d.]+)"\]', (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))[1]
ENVIRONMENT = ROOT / ".venv-newest"
UV = shutil.which("uv")  # makes that environment; without it the gate tests on its own Python alone, and says so


def python_of(environment):
    """The Python of a virtual environment."""
    return environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def on_newest(uv):
    """The checks on the newest Python: its environment made, sift installed in it as CI installs it, the tests."""
    python = str(python_of(ENVIRONMENT))
    return (
        (
            f"an environment of Python {NEWEST}",
            [uv, "venv", "--quiet", "--allow-existing", "--python", NEWEST, str(ENVIRONMENT)],
            10,
        ),
        (
            f"the packages of Python {NEWEST}",
            [uv, "pip", "install", "--quiet", "--python", python, "--torch-backend", "cpu", "-e", f"{ROOT}[dev]"],
            20,
        ),
        (f"the tests on Python {NEWEST}", [python, "-m", "pytest", "-q", "-n", "auto"], 20),
    )


def main(run=subprocess.call, uv=UV, python=sys.version_info[:2]):
    this = ".".join(map(str, python))
    again = on_newest(uv) if uv and this != NEWEST else ()
    for check, command, minutes in (*CHECKS, *again):
        print(f"gate: {check} ...", flush=True)
        try:
            failed = run(command, timeout=60 * minutes)
        except subprocess.TimeoutExpired:
            raise SystemExit(f"gate: {check} took more than {minutes} minutes, and was stopped") from None
        if failed:
            raise SystemExit(f"gate: {check} failed")
    if not uv and this != NEWEST:
        print(f"gate: no uv on the PATH: the tests ran on Python {this} alone, not on {NEWEST}")
    print("gate: green")


if __name__ == "__main__":
    main()
