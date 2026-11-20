"""The gate, `python scripts/gate.py`: the checks a change must pass, in one command. Its checks are fakes here: the
real ones are this very test run, the hooks and the type checker."""

import importlib.util
import sys
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
    gate.main(run=lambda command: ran.append(command) or 0)
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
    types = [command for _, command in gate.CHECKS if "pyright" in command]
    assert [command[5:] for command in types] == [["--pythonpath", sys.executable]] * len(SYSTEMS)


def test_the_gate_ends_at_the_first_check_that_fails(gate, capsys):
    ran = []

    def failing(command):
        ran.append(command)
        return "pytest" in command  # the tests fail

    with pytest.raises(SystemExit, match="gate: the tests, with the coverage gate, failed"):
        gate.main(run=failing)
    assert "pytest" in ran[-1]  # no smoke run of what the tests refuse
    assert "gate: green" not in capsys.readouterr().out


def test_a_check_of_the_types_that_fails_names_its_system(gate):
    with pytest.raises(SystemExit, match="gate: the types as on Darwin failed"):
        gate.main(run=lambda command: "Darwin" in command)


def test_the_checks_are_those_ci_runs(gate):
    # Green here and red there helps nobody: the hooks, the type checker and the tests are called as the workflow
    # calls them, in an environment that is activated there.
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    hooks, *types, tests, _ = (command[2:] for _, command in gate.CHECKS)
    assert f"pre-commit {' '.join(hooks[1:])}" in workflow
    assert f"pytest {' '.join(tests[1:])}" in workflow
    for command in types:
        assert f"run: {' '.join(command[:3])}\n" in workflow
