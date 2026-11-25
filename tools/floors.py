import re
from collections.abc import Callable
from importlib import metadata

from tools.findings import Finding, read, report

FLOOR = re.compile(r'^\s+"([A-Za-z0-9][\w.-]*)>=([0-9]+(?:\.[0-9]+)*)"')
RELEASE = re.compile(r"[0-9]+(?:\.[0-9]+)*")
REMEDY = "Install the lowest versions afresh: nox -s lowest makes its environment anew on each run."


def _release(version: str) -> str:
    found = RELEASE.match(version)
    return re.sub(r"(\.0+)+$", "", found.group() if found else version)


def unmet(project: str, installed: Callable[[str], str]) -> list[Finding]:
    floors = ((number, FLOOR.match(line)) for number, line in enumerate(project.split("\n"), 1))
    return [
        Finding("pyproject.toml", number, f"{name} {installed(name)} is installed, not its lowest version {floor}")
        for number, match in floors
        if match is not None
        for name, floor in [match.groups()]
        if _release(installed(name)) != _release(floor)
    ]


def main(_argv: list[str]) -> int:
    return report(unmet(read("pyproject.toml"), metadata.version), REMEDY)
