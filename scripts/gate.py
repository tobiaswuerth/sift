"""The gate: every check a change must pass, in one command, `python scripts/gate.py`. Green means all passed.

The pre-commit hooks (format, lint, the checks of pyproject.toml and the workflows, spelling), the types of all of
sift, the tests in a process per core with the coverage gate and the rules tests/test_package.py keeps, and
the smoke run of every command as a user runs it (scripts/smoke.py). The first check that fails ends it, and so does
one that takes longer than its minutes: a test that hangs holds nobody for good. CI runs the hooks, the type checker,
the tests and the smoke run the same way, the tests and the smoke run on every system and Python.

It checks the checkout it is in, whichever folder it runs from: its checks run there, on the sift its Python imports,
which must be that checkout's, and on every file there, which git must know of (tracked or ignored), since the hooks
check only the files git tracks.

The types are checked as on each system sift runs on, whichever runs the gate: what only another system has, or
lacks, is found here, not in CI. pyright reads the packages of the Python that runs the gate, which it is told: left
to itself it asks the `python` on the PATH, another one unless the environment is activated.

With uv on the PATH it then builds and checks the package as CI's package job does (scripts/package.py), and the
tests run once more on the newest Python CI tests on, where a Python warns of what an older one lets pass: in an
environment of its own, .venv-newest beside .venv, which uv makes on the first run and brings up to date on each.
The gate owns that folder; deleting it costs the next run its install. Last, as CI's lowest job, the tests run with
the lowest versions sift names of what it depends on, on the oldest Python CI tests on: in an environment made afresh
in a temporary folder each time, since one kept would keep versions above those, and deleted at the end."""

import functools
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
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
# The oldest and the newest Python CI tests on: the first and the last of its matrix
MATRIX = re.search(r"python: \[(.*)\]", (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))[1]
OLDEST, *_, NEWEST = re.findall(r'"([\d.]+)"', MATRIX)
ENVIRONMENT = ROOT / ".venv-newest"
RUN = functools.partial(subprocess.call, cwd=ROOT)  # each check, in the checkout: its tests, its settings
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


def on_lowest(uv, environment):
    """The checks with the lowest versions sift names, on the oldest Python, as CI's lowest job: a new environment,
    those versions in it, which are the ones named (scripts/floors.py), the tests."""
    python = str(python_of(environment))
    lowest = ["--resolution", "lowest-direct", "--torch-backend", "cpu", "-e", f"{ROOT}[dev]"]
    return (
        (f"an environment of Python {OLDEST}", [uv, "venv", "--quiet", "--python", OLDEST, str(environment)], 10),
        ("the lowest versions sift names", [uv, "pip", "install", "--quiet", "--python", python, *lowest], 20),
        ("that those are the versions installed", [python, str(Path(__file__).parent / "floors.py")], 2),
        ("the tests with the lowest versions", [python, "-m", "pytest", "-q", "-n", "auto"], 20),
    )


def sift_at():
    """The folder of the sift the checks import, or None if they find none."""
    spec = importlib.util.find_spec("sift")
    return None if spec is None or spec.origin is None else Path(spec.origin).parent


def untracked(root=ROOT):
    """The files of a checkout git neither tracks nor ignores."""
    listed = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        encoding="utf-8",
        check=True,
        timeout=60,
    )
    return listed.stdout.split("\0")[:-1]


def main(run=RUN, uv=UV, python=sys.version_info[:2]):
    at = sift_at()
    if at is None or at.resolve() != (ROOT / "sift").resolve():
        found = "no sift" if at is None else f"sift from {at}"
        raise SystemExit(
            f"gate: this Python imports {found}, not this checkout's: install it with `uv pip install -e .` here, or "
            f"set PYTHONPATH to {ROOT}"
        )
    if new := untracked():
        raise SystemExit(f"gate: the hooks check only the files git knows; add or ignore these first: {' '.join(new)}")
    this = ".".join(map(str, python))
    package = (("the package", [sys.executable, str(Path(__file__).parent / "package.py")], 15),) if uv else ()
    again = on_newest(uv) if uv and this != NEWEST else ()
    with tempfile.TemporaryDirectory(prefix="sift-gate-") as folder:
        lowest = on_lowest(uv, Path(folder) / "lowest") if uv else ()
        for check, command, minutes in (*CHECKS, *package, *again, *lowest):
            print(f"gate: {check} ...", flush=True)
            try:
                failed = run(command, timeout=60 * minutes)
            except subprocess.TimeoutExpired:
                raise SystemExit(f"gate: {check} took more than {minutes} minutes, and was stopped") from None
            if failed:
                raise SystemExit(f"gate: {check} failed")
    if not uv:
        newest = "" if this == NEWEST else f", the tests ran on Python {this} alone, not on {NEWEST}"
        print(f"gate: no uv on the PATH: the package and the lowest versions sift names went unchecked{newest}")
    print("gate: green")


if __name__ == "__main__":
    main()
