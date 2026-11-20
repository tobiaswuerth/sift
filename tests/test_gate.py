"""The gate, `python scripts/gate.py`: the checks a change must pass, in one command. Its checks are fakes here: the
real ones are this very test run, and the hooks."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location("gate", ROOT / "scripts" / "gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gate_runs_the_hooks_the_tests_and_the_smoke_run(gate, capsys):
    ran = []
    gate.main(run=lambda command: ran.append(command) or 0)
    assert [command[:3] for command in ran] == [
        [sys.executable, "-m", "pre_commit"],
        [sys.executable, "-m", "pytest"],
        [sys.executable, str(ROOT / "scripts" / "smoke.py")],
    ]
    assert "--all-files" in ran[0]  # every file, as CI checks them, not only those changed
    assert "--cov" in ran[1]  # with the coverage gate
    assert capsys.readouterr().out.strip().endswith("gate: green")


def test_the_gate_ends_at_the_first_check_that_fails(gate, capsys):
    ran = []

    def failing(command):
        ran.append(command)
        return len(ran) == 2  # the tests fail

    with pytest.raises(SystemExit, match="gate: the tests, with the coverage gate, failed"):
        gate.main(run=failing)
    assert len(ran) == 2  # no smoke run of what the tests refuse
    assert "gate: green" not in capsys.readouterr().out


def test_the_checks_are_those_ci_runs(gate):
    # Green here and red there helps nobody: the hooks and the tests are called as the workflow calls them.
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    hooks, tests, _ = (" ".join(command[3:]) for _, command in gate.CHECKS)
    assert f"pre-commit {hooks}" in workflow
    assert f"pytest {tests}" in workflow
