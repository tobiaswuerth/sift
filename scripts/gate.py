"""The gate: every check a change must pass, in one command, `python scripts/gate.py`. Green means all passed.

The pre-commit hooks (format, lint, the checks of pyproject.toml and the workflows, spelling), the tests in a process
per core with the coverage gate and the rules tests/test_package.py keeps, and the smoke run of every command as a
user runs it (scripts/smoke.py). The first check that fails ends it. CI runs the hooks and the tests the same way, the
tests on every system and Python."""

import subprocess
import sys
from pathlib import Path

CHECKS = (
    ("the pre-commit hooks", [sys.executable, "-m", "pre_commit", "run", "--all-files", "--show-diff-on-failure"]),
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
