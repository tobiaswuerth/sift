import re
import tomllib
from collections.abc import Callable, Iterator
from importlib import metadata
from typing import cast

from tools.findings import Finding, read, report

PROJECT = "pyproject.toml"
REQUIREMENT = re.compile(r"\s*([A-Za-z0-9][\w.-]*)\s*(?:\[[^\]]*\])?([^;]*)")
FLOOR = re.compile(r">=\s*([0-9]+(?:\.[0-9]+)*)")
RELEASE = re.compile(r"[0-9]+(?:\.[0-9]+)*")
UNREAD = "{requirement!r} names no package: write it as NAME>=VERSION"
NO_FLOOR = "{name} names no lowest version, so none is tested: write {name}>=VERSION"
ABOVE = "{name} {held} is installed, not its lowest version {floor}"
REMEDY = "Install the lowest versions afresh: nox -s lowest makes its environment anew on each run."


def _release(version: str) -> str:
    found = RELEASE.match(version)
    return re.sub(r"(\.0+)+$", "", found.group() if found else version)


def _line(project: str, requirement: str) -> int:
    lines = enumerate(project.split("\n"), 1)
    return next((number for number, line in lines if requirement in line), 1)


def _dependencies(project: str) -> list[str]:
    held = cast("dict[str, dict[str, object]]", tomllib.loads(project))
    return [str(each) for each in cast("list[object]", held["project"]["dependencies"])]


def _said(requirement: str, installed: Callable[[str], str]) -> Iterator[str]:
    found = REQUIREMENT.fullmatch(requirement.partition(";")[0])
    if found is None:
        yield UNREAD.format(requirement=requirement)
        return
    name, floor = found[1], FLOOR.search(found[2])
    if floor is None:
        yield NO_FLOOR.format(name=name)
    elif _release(installed(name)) != _release(floor[1]):
        yield ABOVE.format(name=name, held=installed(name), floor=floor[1])


def unmet(project: str, installed: Callable[[str], str]) -> list[Finding]:
    return [
        Finding(PROJECT, _line(project, requirement), said)
        for requirement in _dependencies(project)
        for said in _said(requirement, installed)
    ]


def main(_argv: list[str]) -> int:
    return report(unmet(read(PROJECT), metadata.version), REMEDY)
