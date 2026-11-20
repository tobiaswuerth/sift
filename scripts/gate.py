"""The gate: every check a change must pass, in one command, `python scripts/gate.py`. Green means all passed.

The pre-commit hooks (format, lint, the checks of pyproject.toml and the workflows, spelling), the types of the modules
pyright checks, the tests in a process per core with the coverage gate and the rules tests/test_package.py keeps, and
the smoke run of every command as a user runs it (scripts/smoke.py). The first check that fails ends it. CI runs the
hooks, the type checker and the tests the same way, the tests on every system and Python.

The types are checked as on each system sift runs on, whichever runs the gate: what only another system has, or
lacks, is found here, not in CI. pyright reads the packages of the Python that runs the gate, which it is told: left
to itself it asks the `python` on the PATH, another one unless the environment is activated."""

import subprocess
import sys
from pathlib import Path

SYSTEMS = ("Linux", "Darwin", "Windows")  # as pyright names them
CHECKS = (
    ("the pre-commit hooks", [sys.executable, "-m", "pre_commit", "run", "--all-files", "--show-diff-on-failure"]),
    *(
        (
            f"the types as on {system}",
            [sys.executable, "-m", "pyright", "--pythonplatform", system, "--pythonpath", sys.executable],
        )
        for system in SYSTEMS
    ),
    ("the tests, with the coverage gate,", [sys.executable, "-m", "pytest", "-q", "--cov", "-n", "auto"]),
    ("the smoke run", [sys.executable, str(Path(__file__).parent / "smoke.py")]),
)


def main(run=subprocess.call):
    for check, command in CHECKS:
        print(f"gate: {check} ...", flush=True)
        if run(command):
            raise SystemExit(f"gate: {check} failed")
    print("gate: green")


if __name__ == "__main__":
    main()
