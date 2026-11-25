import ast
import tomllib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import cast

from tools.canonical import aliases, imports_resolved, unquoted
from tools.findings import Finding, read, report, sources

SOURCE = "src/sift"
SURFACE = "sift.api"
SCRIPT = "__main__"
PROTOCOLS = frozenset({"typing.Protocol", "typing_extensions.Protocol"})
REMEDY = "Delete what nothing uses, or use it: a test alone keeps no code alive."


def module_of(path: str) -> str:
    parts = PurePosixPath(path).with_suffix("").parts[1:]
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


@dataclass(frozen=True, slots=True)
class _Module:
    path: str
    name: str
    tree: ast.Module


@dataclass(frozen=True, slots=True)
class _Known:
    modules: frozenset[str]
    protocols: Mapping[str, frozenset[str]]


def _defined(body: Iterable[ast.stmt]) -> Iterator[tuple[str, ast.stmt]]:
    for node in body:
        match node:
            case (
                ast.FunctionDef(name=name)
                | ast.AsyncFunctionDef(name=name)
                | ast.ClassDef(name=name)
                | ast.TypeAlias(name=ast.Name(id=name))
                | ast.AnnAssign(target=ast.Name(id=name))
            ):
                yield name, node
            case ast.Assign(targets=targets):
                yield from ((each.id, node) for each in targets if isinstance(each, ast.Name))
            case _:
                pass


def _qualified(module: str, node: ast.expr) -> str:
    match node:
        case ast.Name(id=name):
            return f"{module}.{name}"
        case ast.Subscript(value=value):
            return _qualified(module, value)
        case _:
            return ast.unparse(node)


def _protocols(modules: Iterable[_Module]) -> Mapping[str, frozenset[str]]:
    classes = {
        f"{module.name}.{node.name}": (module.name, node)
        for module in modules
        for node in module.tree.body
        if isinstance(node, ast.ClassDef) and any(_qualified(module.name, base) in PROTOCOLS for base in node.bases)
    }

    def members(protocol: str) -> frozenset[str]:
        module, node = classes[protocol]
        bases = (_qualified(module, base) for base in node.bases)
        own = frozenset(name for name, _ in _defined(node.body))
        return own.union(*(members(base) for base in bases if base in classes))

    return MappingProxyType({protocol: members(protocol) for protocol in classes})


def _references(module: str, node: ast.AST, known: _Known) -> Iterator[str]:
    for each in ast.walk(node):
        match each:
            case ast.ImportFrom(module=str(source), names=names, level=0):
                yield source
                yield from (whole for alias in names if (whole := f"{source}.{alias.name}") in known.modules)
            case ast.Import(names=names):
                yield from (alias.name for alias in names)
            case ast.AnnAssign(annotation=annotation, value=ast.expr() as value) if (
                protocol := _qualified(module, annotation)
            ) in known.protocols:
                yield from (f"{ast.unparse(value)}.{member}" for member in known.protocols[protocol])
            case ast.Attribute():
                yield ast.unparse(each)
            case ast.Name(id=name, ctx=ast.Load()):
                yield f"{module}.{name}"
            case _:
                pass


def _edges(module: _Module, known: _Known) -> Iterator[tuple[str, list[str]]]:
    defined = list(_defined(module.tree.body))
    held = {id(node) for _, node in defined}
    statements = (node for node in module.tree.body if id(node) not in held)
    parent = module.name.rpartition(".")[0]
    yield module.name, [parent, *(each for node in statements for each in _references(module.name, node, known))]
    for name, node in defined:
        yield f"{module.name}.{name}", [module.name, *_references(module.name, node, known)]
    for alias, target in aliases(module.tree).items():
        yield f"{module.name}.{alias}", [target]


def _target(reference: str, nodes: Mapping[str, object]) -> str | None:
    parts = reference.split(".")
    return next(
        (whole for end in range(len(parts), 0, -1) if (whole := ".".join(parts[:end])) in nodes),
        None,
    )


def _reached(edges: Mapping[str, list[str]], roots: Iterable[str]) -> set[str]:
    reached: set[str] = set()
    waiting = [found for root in roots if (found := _target(root, edges)) is not None]
    while waiting:
        node = waiting.pop()
        if node not in reached:
            reached.add(node)
            waiting += [found for each in edges[node] if (found := _target(each, edges)) is not None]
    return reached


def _surface(name: str) -> bool:
    return name == SURFACE or name.startswith(f"{SURFACE}.")


def _kept(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def entry_points(pyproject: str) -> frozenset[str]:
    project = cast("dict[str, dict[str, str]]", tomllib.loads(pyproject).get("project", {}))
    return frozenset(target.replace(":", ".") for target in project.get("scripts", {}).values())


def dead(texts: Mapping[str, str], kept: frozenset[str]) -> tuple[Finding, ...]:
    modules = [
        _Module(path, module_of(path), imports_resolved(unquoted(ast.parse(text)))) for path, text in texts.items()
    ]
    known = _Known(frozenset(module.name for module in modules), _protocols(modules))
    edges: dict[str, list[str]] = {}
    for module in modules:
        for node, references in _edges(module, known):
            edges.setdefault(node, []).extend(references)
    scripts = (module.name for module in modules if module.name.rpartition(".")[2] == SCRIPT)
    reached = _reached(edges, [*kept, *scripts, *(node for node in edges if _surface(node))])
    return tuple(
        Finding(module.path, node.lineno, f"{name} is reached from no entry point, script or the API")
        for module in modules
        if not _surface(module.name)
        for name, node in _defined(module.tree.body)
        if not _kept(name) and f"{module.name}.{name}" not in reached
    )


def main(_argv: list[str]) -> int:
    kept = entry_points(read("pyproject.toml"))
    return report(dead(sources((SOURCE,)), kept), REMEDY)
