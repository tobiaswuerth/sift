import ast
import copy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import cast, final, override

from tools.canonical import imports_resolved
from tools.findings import PYTHON_ROOTS, Finding, report, sources

FEWEST_NODES = 10
BUILTINS = "builtins"
LAMBDA = "lambda"
REMEDY = (
    "Keep one where both places can import it, and call it from the other: a fix made in one twin misses the other."
)


@dataclass(frozen=True, slots=True)
class Function:
    name: str
    line: int
    arguments: ast.arguments
    body: list[ast.stmt]
    decorators: list[ast.expr]
    module: Mapping[str, str]


@final
class _Canonical(ast.NodeTransformer):
    def __init__(self, own: frozenset[str], module: Mapping[str, str]) -> None:
        self.own, self.module = own, module
        self.given: dict[str, str] = {}

    def _name(self, name: str) -> str:
        if name in self.own:
            return self.given.setdefault(name, f"_{len(self.given)}")
        return self.module.get(name, name)

    @override
    def visit_Name(self, node: ast.Name) -> ast.Name:
        return ast.Name(self._name(node.id), node.ctx)

    @override
    def visit_arg(self, node: ast.arg) -> ast.arg:
        return ast.arg(self._name(node.arg))

    @override
    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == BUILTINS:
            return ast.Name(node.attr, node.ctx)
        return node

    @override
    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        node.keywords.sort(key=lambda keyword: (keyword.arg is None, keyword.arg or ""))
        return node


def _own_names(function: Function) -> frozenset[str]:
    parts = [function.arguments, *function.body]
    stored = (
        node.id
        for part in parts
        for node in ast.walk(part)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    return frozenset((*(each.arg for each in ast.walk(function.arguments) if isinstance(each, ast.arg)), *stored))


def _size(function: Function) -> int:
    return sum(isinstance(node, ast.stmt | ast.expr) for statement in function.body for node in ast.walk(statement))


def shape_of(function: Function) -> str:
    canonical = _Canonical(_own_names(function), function.module)
    arguments = cast("ast.arguments", canonical.visit(copy.deepcopy(function.arguments)))
    arguments.defaults, arguments.kw_defaults = [], []
    body = [cast("ast.stmt", canonical.visit(each)) for each in copy.deepcopy(function.body)]
    decorators = [cast("ast.expr", canonical.visit(each)) for each in copy.deepcopy(function.decorators)]
    return "\n".join(ast.dump(part) for part in (arguments, ast.Module(body, []), ast.List(decorators)))


def _function(node: ast.AST, module: Mapping[str, str]) -> Function | None:
    match node:
        case ast.FunctionDef() | ast.AsyncFunctionDef():
            others = {name: whole for name, whole in module.items() if name != node.name}
            return Function(node.name, node.lineno, node.args, node.body, node.decorator_list, others)
        case ast.Lambda(args=arguments, body=body):
            return Function(LAMBDA, node.lineno, arguments, [ast.Return(body)], [], module)
        case _:
            return None


def _bound(node: ast.stmt) -> Iterator[str]:
    match node:
        case (
            ast.FunctionDef(name=name)
            | ast.AsyncFunctionDef(name=name)
            | ast.ClassDef(name=name)
            | ast.AnnAssign(target=ast.Name(id=name))
        ):
            yield name
        case ast.Assign(targets=targets):
            yield from (each.id for each in targets if isinstance(each, ast.Name))
        case _:
            pass


def _defined(path: str, tree: ast.Module) -> Mapping[str, str]:
    return {name: f"{path}:{name}" for node in tree.body for name in _bound(node)}


def _functions(texts: Mapping[str, str]) -> Iterator[tuple[str, Function]]:
    for path, text in texts.items():
        tree = imports_resolved(ast.parse(text))
        module = _defined(path, tree)
        for node in ast.walk(tree):
            function = _function(node, module)
            if function is not None and _size(function) >= FEWEST_NODES:
                yield path, function


def twins(texts: Mapping[str, str]) -> tuple[Finding, ...]:
    first: dict[str, str] = {}
    found: list[Finding] = []
    for path, function in _functions(texts):
        shape = shape_of(function)
        if shape in first:
            found.append(Finding(path, function.line, f"{function.name} does what {first[shape]} does"))
        else:
            first[shape] = f"{function.name} ({path}:{function.line})"
    return tuple(found)


def main(_argv: list[str]) -> int:
    return report(twins(sources(PYTHON_ROOTS)), REMEDY)
