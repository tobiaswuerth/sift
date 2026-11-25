import ast
import re
import sys
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from importlib import metadata
from typing import cast

from tools.findings import Finding, read, report, sources

SOURCE = "src/sift"
OWN = frozenset({"sift", "__future__"})
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
REMEDY = "Name the distribution in [project] dependencies of pyproject.toml: sift's users get only what it names."
LATE_REMEDY = (
    "Under if TYPE_CHECKING:, import the module (from sift.app import retitling as _retitling) and assign it "
    "to its protocol (_RETITLING: Retitling = _retitling): pyright then proves it fits, and the import contracts "
    "see the edge."
)
LOADERS = frozenset({"late", "import_module"})
OWN_PREFIX = "sift."
UNPROVEN = "loads {module} by name without importing it under TYPE_CHECKING and assigning it to its protocol"


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


def _constants(tree: ast.Module) -> dict[str, str]:
    return {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _named(argument: ast.expr, constants: Mapping[str, str]) -> str | None:
    match argument:
        case ast.Constant(value=str(text)):
            return text
        case ast.Name(id=name):
            return constants.get(name)
        case _:
            return None


def _loaded(tree: ast.Module) -> Iterator[tuple[int, str]]:
    constants = _constants(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in LOADERS and node.args:
            module = _named(node.args[0], constants)
            if module is not None and module.startswith(OWN_PREFIX):
                yield node.lineno, module


def _guarded(tree: ast.Module) -> Iterator[ast.stmt]:
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            yield from node.body


def _aliases(statement: ast.stmt) -> Iterator[tuple[str, str]]:
    match statement:
        case ast.ImportFrom(module=str(package), names=names, level=0):
            yield from ((alias.asname or alias.name, f"{package}.{alias.name}") for alias in names)
        case ast.Import(names=names):
            yield from ((alias.asname, alias.name) for alias in names if alias.asname)
        case _:
            pass


def _proven(tree: ast.Module) -> frozenset[str]:
    guarded = list(_guarded(tree))
    modules = dict(alias for statement in guarded for alias in _aliases(statement))
    assigned = {
        statement.value.id
        for statement in guarded
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.value, ast.Name)
    }
    return frozenset(module for alias, module in modules.items() if alias in assigned)


def late_loaded(files: Mapping[str, str]) -> list[Finding]:
    found: list[Finding] = []
    for path, text in files.items():
        tree = ast.parse(text)
        proven = _proven(tree)
        found += [
            Finding(path, line, UNPROVEN.format(module=module))
            for line, module in _loaded(tree)
            if module not in proven
        ]
    return found


def main(_argv: list[str]) -> int:
    depended = declared(read("pyproject.toml"))
    held = sources((SOURCE,))
    undeclared_found = report(undeclared(held, metadata.packages_distributions(), depended), REMEDY)
    return max(undeclared_found, report(late_loaded(held), LATE_REMEDY))
