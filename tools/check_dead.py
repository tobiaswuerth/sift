import ast
import tomllib
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import cast

from tools.canonical import imports_resolved
from tools.findings import Finding, report, sources

SOURCE = "src/sift"
SURFACE = "src/sift/api/"
REMEDY = "Delete what nothing uses, or use it: a test alone keeps no code alive."


def module_of(path: str) -> str:
    parts = PurePosixPath(path).with_suffix("").parts[1:]
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _defined(tree: ast.Module) -> Iterator[tuple[str, int]]:
    for node in tree.body:
        match node:
            case ast.FunctionDef(name=name) | ast.ClassDef(name=name) | ast.TypeAlias(name=ast.Name(id=name)):
                yield name, node.lineno
            case ast.Assign(targets=targets):
                yield from ((each.id, node.lineno) for each in targets if isinstance(each, ast.Name))
            case ast.AnnAssign(target=ast.Name(id=name)):
                yield name, node.lineno
            case _:
                pass


def _protocol_members(tree: ast.Module) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(ast.unparse(base).endswith("Protocol") for base in node.bases):
            yield from (name for name, _ in _defined(ast.Module(body=node.body, type_ignores=[])))


def _references(module: str, tree: ast.Module) -> Iterator[str]:
    for node in ast.walk(imports_resolved(tree)):
        match node:
            case ast.ImportFrom(module=str(source), names=names, level=0):
                yield from (f"{source}.{alias.name}" for alias in names)
            case ast.Attribute():
                yield ast.unparse(node)
            case ast.Name(id=name, ctx=ast.Load()):
                yield f"{module}.{name}"
            case _:
                pass


def entry_points(pyproject: str) -> frozenset[str]:
    project = cast("dict[str, dict[str, str]]", tomllib.loads(pyproject).get("project", {}))
    return frozenset(target.replace(":", ".") for target in project.get("scripts", {}).values())


def dead(sources: Mapping[str, str], kept: frozenset[str]) -> tuple[Finding, ...]:
    trees = {path: ast.parse(source) for path, source in sources.items()}
    used = {reference for path, tree in trees.items() for reference in _references(module_of(path), tree)}
    members = {name for tree in trees.values() for name in _protocol_members(tree)}
    return tuple(
        Finding(path, line, f"{name} is used nowhere in sift")
        for path, tree in trees.items()
        if not path.startswith(SURFACE)
        for name, line in _defined(tree)
        if not name.startswith("_")
        and name not in members
        and (whole := f"{module_of(path)}.{name}") not in used
        and whole not in kept
    )


def main(_argv: list[str]) -> int:
    kept = entry_points(Path("pyproject.toml").read_text(encoding="utf-8"))
    return report(dead(sources((SOURCE,)), kept), REMEDY)
