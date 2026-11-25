import pytest

from tools.findings import Finding
from tools.floors import main, unmet

PROJECT = """[project]
dependencies = [
    "numpy>=2.0",  # why
    "torch>=2.13",  # why
    "starlette>=1.3.1",  # why
    "uvicorn",
]
"""


def test_each_dependency_installed_at_its_lowest_version_is_met() -> None:
    held = {"numpy": "2.0.0", "torch": "2.13.0+cpu", "starlette": "1.3.1", "uvicorn": "0.54.0"}
    assert unmet(PROJECT, held.__getitem__) == []


def test_a_dependency_installed_above_its_lowest_version_is_named_with_its_line() -> None:
    held = {"numpy": "2.0.1", "torch": "2.15.0+cpu", "starlette": "1.3.1", "uvicorn": "0.54.0"}
    assert unmet(PROJECT, held.__getitem__) == [
        Finding("pyproject.toml", 3, "numpy 2.0.1 is installed, not its lowest version 2.0"),
        Finding("pyproject.toml", 4, "torch 2.15.0+cpu is installed, not its lowest version 2.13"),
    ]


def test_the_tool_reads_this_project_and_what_this_environment_holds(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([])
    said = capsys.readouterr().out
    assert ("is installed, not its lowest version" in said, code) in {(False, 0), (True, 1)}
