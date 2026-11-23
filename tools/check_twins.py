import ast
import copy
from collections.abc import Iterator, Mapping
from typing import cast, final, override

from tools.findings import PYTHON_ROOTS, Finding, report, sources

FEWEST_NODES = 10
REMEDY = (
    "Keep one where both places can import it, and call it from the other: a fix made in one twin misses the other."
)


@final
class _Renamed(ast.NodeTransformer):
    def __init__(self, own: frozenset[str]) -> None:
        self.own = own
        self.given: dict[str, str] = {}

    def _name(self, name: str) -> str:
        return self.given.setdefault(name, f"_{len(self.given)}") if name in self.own else name

    @override
    def visit_Name(self, node: ast.Name) -> ast.Name:
        return ast.Name(self._name(node.id), node.ctx)

    @override
    def visit_arg(self, node: ast.arg) -> ast.arg:
        return ast.arg(self._name(node.arg))


def _own_names(function: ast.FunctionDef) -> frozenset[str]:
    stored = (node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
    return frozenset((*(each.arg for each in ast.walk(function.args) if isinstance(each, ast.arg)), *stored))


def _size(function: ast.FunctionDef) -> int:
    return sum(isinstance(node, ast.stmt | ast.expr) for statement in function.body for node in ast.walk(statement))


def shape_of(function: ast.FunctionDef) -> str:
    renamed = cast("ast.FunctionDef", _Renamed(_own_names(function)).visit(copy.deepcopy(function)))
    renamed.name, renamed.returns = "", None
    return ast.dump(renamed)


def _functions(sources: Mapping[str, str]) -> Iterator[tuple[str, ast.FunctionDef]]:
    for path, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and _size(node) >= FEWEST_NODES:
                yield path, node


def twins(sources: Mapping[str, str]) -> tuple[Finding, ...]:
    first: dict[str, str] = {}
    found: list[Finding] = []
    for path, function in _functions(sources):
        shape = shape_of(function)
        if shape in first:
            found.append(Finding(path, function.lineno, f"{function.name} does what {first[shape]} does"))
        else:
            first[shape] = f"{function.name} ({path}:{function.lineno})"
    return tuple(found)


def main(_argv: list[str]) -> int:
    return report(twins(sources(PYTHON_ROOTS)), REMEDY)
