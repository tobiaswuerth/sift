import pytest

from tools.findings import Finding
from tools.floors import main, unmet

PROJECT = """[project]
dependencies = [
    "numpy>=2.0",  # why
    "torch >= 2.13, <3",  # why
    "starlette[full]>=1.3.1",  # why
    "uvicorn>=0.31; python_version >= '3.12'",
]
"""


def test_each_dependency_installed_at_its_lowest_version_is_met_however_it_is_written() -> None:
    held = {"numpy": "2.0.0", "torch": "2.13.0+cpu", "starlette": "1.3.1", "uvicorn": "0.31"}
    assert unmet(PROJECT, held.__getitem__) == []
    assert unmet('[project]\ndependencies = ["numpy>=2.0", "torch>=2.13"]\n', held.__getitem__) == []


def test_a_dependency_installed_above_its_lowest_version_is_named_with_its_line() -> None:
    held = {"numpy": "2.0.1", "torch": "2.15.0+cpu", "starlette": "1.3.1", "uvicorn": "0.54.0"}
    assert unmet(PROJECT, held.__getitem__) == [
        Finding("pyproject.toml", 3, "numpy 2.0.1 is installed, not its lowest version 2.0"),
        Finding("pyproject.toml", 4, "torch 2.15.0+cpu is installed, not its lowest version 2.13"),
        Finding("pyproject.toml", 6, "uvicorn 0.54.0 is installed, not its lowest version 0.31"),
    ]


def test_a_dependency_without_a_lowest_version_or_that_names_no_package_is_a_finding_not_passed_over() -> None:
    project = '[project]\ndependencies = [\n    "uvicorn",\n    "torch<3",\n    "@ https://example.com/x.whl",\n]\n'
    assert unmet(project, {"uvicorn": "0.54.0", "torch": "2.13"}.__getitem__) == [
        Finding("pyproject.toml", 3, "uvicorn names no lowest version, so none is tested: write uvicorn>=VERSION"),
        Finding("pyproject.toml", 4, "torch names no lowest version, so none is tested: write torch>=VERSION"),
        Finding("pyproject.toml", 5, "'@ https://example.com/x.whl' names no package: write it as NAME>=VERSION"),
    ]


def test_the_tool_reads_this_project_whose_every_dependency_names_a_lowest_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([])
    said = capsys.readouterr().out
    assert ("names no lowest version" in said, "names no package" in said) == (False, False)
    assert ("is installed, not its lowest version" in said) == (code == 1)
