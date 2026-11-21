"""The gate, `python scripts/gate.py`: the checks a change must pass, in one command. Its checks are fakes here: the
real ones are this very test run, the hooks and the type checker."""

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SYSTEMS = ["Linux", "Darwin", "Windows"]  # as pyright names those sift runs on


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location("gate", ROOT / "scripts" / "gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gate_runs_the_hooks_the_types_the_tests_and_the_smoke_run(gate, capsys):
    ran = []
    gate.main(run=lambda command, timeout: ran.append(command) or 0, uv=None)
    hooks, *types, tests, smoke = ran
    assert hooks[:3] == [sys.executable, "-m", "pre_commit"]
    assert "--all-files" in hooks  # every file, as CI checks them, not only those changed
    # The types as on every system sift runs on, whichever runs the gate: code for another is checked here too.
    assert [command[:5] for command in types] == [
        [sys.executable, "-m", "pyright", "--pythonplatform", system] for system in SYSTEMS
    ]
    assert tests[:3] == [sys.executable, "-m", "pytest"]
    assert "--cov" in tests  # with the coverage gate
    assert smoke == [sys.executable, str(ROOT / "scripts" / "smoke.py")]
    assert capsys.readouterr().out.strip().endswith("gate: green")


def test_the_types_are_checked_against_the_packages_of_the_gate_s_python(gate):
    # pyright asks the `python` on the PATH for the packages sift imports: run without its environment activated, the
    # gate had it read another Python's, or none.
    types = [command for _, command, _ in gate.CHECKS if "pyright" in command]
    assert [command[5:] for command in types] == [["--pythonpath", sys.executable]] * len(SYSTEMS)


def test_the_gate_ends_at_the_first_check_that_fails(gate, capsys):
    ran = []

    def failing(command, timeout):
        ran.append(command)
        return "pytest" in command  # the tests fail

    with pytest.raises(SystemExit, match="gate: the tests, with the coverage gate, failed"):
        gate.main(run=failing)
    assert "pytest" in ran[-1]  # no smoke run of what the tests refuse
    assert "gate: green" not in capsys.readouterr().out


def test_a_check_of_the_types_that_fails_names_its_system(gate):
    with pytest.raises(SystemExit, match="gate: the types as on Darwin failed"):
        gate.main(run=lambda command, timeout: "Darwin" in command)


def test_a_check_that_comes_to_no_end_is_stopped(gate):
    # No check had a time to end in: a test that hung held the gate, and whoever ran it, for good.
    limits = []

    def hanging(command, timeout):
        limits.append(timeout)
        if "pytest" in command:
            raise subprocess.TimeoutExpired(command, timeout)
        return 0

    with pytest.raises(SystemExit, match="gate: the tests, with the coverage gate, took more than 20 minutes"):
        gate.main(run=hanging)
    assert limits[-1] == 20 * 60
    assert all(limit > 0 for limit in limits)


def test_a_test_that_hangs_says_where_before_the_gate_stops_it(gate):
    # Stopped from outside, a hung test said nothing of where it stood: pytest writes every thread's stack once a
    # test takes this long, well before the gate's patience with all of them ends.
    options = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]
    (minutes,) = [minutes for _, command, minutes in gate.CHECKS if "pytest" in command]
    assert 0 < options["faulthandler_timeout"] <= 60 * minutes / 2


def test_the_pull_request_template_asks_for_the_gate():
    # It asked for the tests and the hooks, each by its command: the types and the smoke run, which the gate added,
    # went unasked.
    template = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    assert "`python scripts/gate.py`" in template
    assert not {"pytest", "pre-commit", "pyright"} & set(template.replace("`", " ").split()), (
        "ask for the gate, which runs every check, not for some of its checks by their commands"
    )


def test_the_checks_are_those_ci_runs(gate):
    # Green here and red there helps nobody: the hooks, the type checker and the tests are called as the workflow
    # calls them, in an environment that is activated there.
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    hooks, *types, tests, _ = (command[2:] for _, command, _ in gate.CHECKS)
    assert f"pre-commit {' '.join(hooks[1:])}" in workflow
    assert f"pytest {' '.join(tests[1:])}" in workflow
    assert "run: python scripts/smoke.py" in workflow  # on every system: it ran on the one that ran the gate alone
    for command in types:
        assert f"run: {' '.join(command[:3])}\n" in workflow


def test_the_gate_tests_on_the_newest_python_too(gate, capsys):
    # The gate ran on one Python, and CI on five: a test that left a database open, a warning on Python 3.13 and
    # newer, passed here and failed there. The tests run again on the newest Python CI has, in an environment of its
    # own that uv makes and keeps.
    ran = []
    gate.main(run=lambda command, timeout: ran.append(command) or 0, uv="uv", python=(3, 12))
    made, installed, tests = ran[-3:]
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert f', "{gate.NEWEST}"]' in workflow  # the last Python of CI's matrix
    assert made == ["uv", "venv", "--quiet", "--allow-existing", "--python", gate.NEWEST, str(gate.ENVIRONMENT)]
    there = str(gate.python_of(gate.ENVIRONMENT))
    assert installed[:5] == ["uv", "pip", "install", "--quiet", "--python"]
    assert installed[5:] == [there, "--torch-backend", "cpu", "-e", f"{ROOT}[dev]"]  # as CI installs sift
    assert tests == [there, "-m", "pytest", "-q", "-n", "auto"]
    assert ".venv-newest/" in (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert capsys.readouterr().out.strip().endswith("gate: green")


def test_a_gate_on_the_newest_python_tests_once(gate):
    ran = []
    newest = tuple(map(int, gate.NEWEST.split(".")))
    gate.main(run=lambda command, timeout: ran.append(command) or 0, uv="uv", python=newest)
    assert len(ran) == len(gate.CHECKS) + 1  # and the package


def test_the_gate_checks_the_package_as_ci_does(gate):
    # CI's package job built the package and checked its changelog, its metadata and its wheel; the gate didn't, and
    # a version without its changelog section passed here and failed there.
    ran = []
    gate.main(run=lambda command, timeout: ran.append(command) or 0, uv="uv", python=(3, 12))
    assert ran[len(gate.CHECKS)] == [sys.executable, str(ROOT / "scripts" / "package.py")]
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "run: uv run --no-project scripts/package.py" in workflow


def test_a_gate_without_uv_says_which_python_it_tested_on(gate, capsys):
    # uv makes the newest Python's environment: without it the gate is the checks of this Python, and says so.
    ran = []
    gate.main(run=lambda command, timeout: ran.append(command) or 0, uv=None, python=(3, 12))
    assert len(ran) == len(gate.CHECKS)
    said = capsys.readouterr().out
    assert "gate: no uv on the PATH: the package wasn't built and checked, the tests ran on Python 3.12 alone, " in said
    assert said.strip().endswith("gate: green")
