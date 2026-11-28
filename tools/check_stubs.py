import ast
import builtins
import importlib
import inspect
from collections.abc import Callable, Iterator, Mapping
from pathlib import PurePosixPath
from typing import cast

from tools.findings import Finding, files, read, report

STUBS = "typings"
REMEDY = "Make the stub say what the installed package has: a name the package lacks hides an error until it runs."


def module_of(path: str) -> str:
    parts = PurePosixPath(path).relative_to(STUBS).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _defined(body: list[ast.stmt]) -> Iterator[tuple[ast.stmt, str]]:
    for node in body:
        match node:
            case ast.ClassDef(name=name) | ast.FunctionDef(name=name) | ast.AnnAssign(target=ast.Name(id=name)):
                yield node, name
            case ast.Assign(targets=[ast.Name(id=name)]):
                yield node, name
            case _:
                pass


def _class(path: str, node: ast.ClassDef, actual: object, qualified: str) -> Iterator[Finding]:
    if not inspect.isclass(actual):
        yield Finding(path, node.lineno, f"names the class {qualified}, which is no class")
        return
    named: list[object] = [getattr(builtins, base.id, None) for base in node.bases if isinstance(base, ast.Name)]
    yield from (
        Finding(path, node.lineno, f"says {qualified} is a {base.__name__}, which it is not")
        for base in named
        if isinstance(base, type) and not issubclass(actual, base)
    )
    yield from _missing(path, node.body, actual, qualified)


def _missing(path: str, body: list[ast.stmt], owner: object, prefix: str) -> Iterator[Finding]:
    for node, name in _defined(body):
        qualified = f"{prefix}.{name}"
        if not hasattr(owner, name):
            yield Finding(path, node.lineno, f"names {qualified}, which the installed package lacks")
        elif isinstance(node, ast.ClassDef):
            yield from _class(path, node, cast("object", getattr(owner, name)), qualified)


def missing(stubs: Mapping[str, str], imported: Callable[[str], object]) -> list[Finding]:
    return [
        finding
        for path, text in stubs.items()
        for finding in _missing(path, ast.parse(text).body, imported(module_of(path)), module_of(path))
    ]


def main(_argv: list[str]) -> int:
    stubs = {name: read(name) for name in files((STUBS,), (".pyi",))}
    return report(missing(stubs, importlib.import_module), REMEDY)
