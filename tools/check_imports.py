import ast
import re
import sys
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import cast

from tools.findings import Finding, report, sources

SOURCE = "src/sift"
OWN = frozenset({"sift", "__future__"})
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
REMEDY = "Name the distribution in [project] dependencies of pyproject.toml: sift's users get only what it names."


def normalised(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def declared(pyproject: str) -> frozenset[str]:
    project = cast("dict[str, list[str]]", tomllib.loads(pyproject).get("project", {}))
    names = (NAME.match(each) for each in project.get("dependencies", []))
    return frozenset(normalised(found.group()) for found in names if found)


def _imported(tree: ast.Module) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                yield from ((node.lineno, alias.name.split(".")[0]) for alias in names)
            case ast.ImportFrom(module=str(module), level=0):
                yield node.lineno, module.split(".")[0]
            case _:
                pass


def _problem(module: str, providers: Mapping[str, Sequence[str]], depended: frozenset[str]) -> str | None:
    given = [normalised(each) for each in providers.get(module, ())]
    if not given:
        return f"imports {module}, which no installed distribution provides"
    if depended.isdisjoint(given):
        return f"imports {module}, of {', '.join(given)}, which pyproject.toml does not depend on"
    return None


def undeclared(
    files: Mapping[str, str], providers: Mapping[str, Sequence[str]], depended: frozenset[str]
) -> list[Finding]:
    return [
        Finding(path, line, problem)
        for path, text in files.items()
        for line, module in _imported(ast.parse(text))
        if module not in OWN and module not in sys.stdlib_module_names
        if (problem := _problem(module, providers, depended)) is not None
    ]


def main(_argv: list[str]) -> int:
    depended = declared(Path("pyproject.toml").read_text(encoding="utf-8"))
    return report(undeclared(sources((SOURCE,)), metadata.packages_distributions(), depended), REMEDY)
