import ast
import copy
from collections import Counter
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast, final, override

ATTRIBUTE_GETTERS = frozenset({"getattr", "builtins.getattr"})
CASTS = frozenset({"cast", "typing.cast"})
ANNOTATED = "Annotated"
LITERAL = "Literal"


def aliases(tree: ast.Module) -> Mapping[str, str]:
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update((alias.asname, alias.name) for alias in node.names if alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.update((alias.asname or alias.name, f"{node.module}.{alias.name}") for alias in node.names)
    return MappingProxyType(found)


def _dotted(node: ast.expr) -> bool:
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def bindings(tree: ast.Module) -> Mapping[str, str]:
    assigned = [node for node in ast.walk(tree) if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign)]
    stored = Counter(
        each.id
        for node in assigned
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        for each in ast.walk(target)
        if isinstance(each, ast.Name)
    )
    return MappingProxyType(
        {
            target.id: ast.unparse(node.value)
            for node in assigned
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            for target in node.targets
            if isinstance(target, ast.Name) and stored[target.id] == 1 and _dotted(node.value)
        }
    )


@final
class _Resolved(ast.NodeTransformer):
    def __init__(self, names: Mapping[str, str]) -> None:
        self.names = names
        self.resolving: set[str] = set()

    @override
    def visit_Name(self, node: ast.Name) -> ast.expr:
        dotted = self.names.get(node.id)
        if dotted is None or not isinstance(node.ctx, ast.Load) or node.id in self.resolving:
            return node
        self.resolving.add(node.id)
        resolved = cast("ast.expr", self.visit(ast.parse(dotted, mode="eval").body))
        self.resolving.discard(node.id)
        return ast.copy_location(resolved, node)

    @override
    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        match node:
            case ast.Call(args=[owner, ast.Constant(value=str(attribute))]) if (
                ast.unparse(node.func) in ATTRIBUTE_GETTERS and attribute.isidentifier()
            ):
                return ast.copy_location(ast.Attribute(owner, attribute, ast.Load()), node)
            case _:
                return node


def _last(node: ast.expr) -> str:
    return ast.unparse(node).rpartition(".")[2]


def _type_expression(node: ast.expr) -> ast.expr:
    match node:
        case ast.Constant(value=str(text)):
            return _type_expression(ast.parse(text, mode="eval").body)
        case ast.Subscript(value=value, slice=ast.Tuple(elts=[first, *_]) as metadata) if _last(value) == ANNOTATED:
            metadata.elts[0] = _type_expression(first)
        case ast.Subscript(value=value, slice=part) if _last(value) != LITERAL:
            node.value, node.slice = _type_expression(value), _type_expression(part)
        case ast.Tuple(elts=elements) | ast.List(elts=elements):
            node.elts = [_type_expression(each) for each in elements]
        case ast.BinOp(left=left, right=right):
            node.left, node.right = _type_expression(left), _type_expression(right)
        case _:
            pass
    return node


@final
class _Unquoted(ast.NodeTransformer):
    @override
    def visit_arg(self, node: ast.arg) -> ast.arg:
        if node.annotation is not None:
            node.annotation = _type_expression(node.annotation)
        return node

    def _returning[F: (ast.FunctionDef, ast.AsyncFunctionDef)](self, node: F) -> F:
        if node.returns is not None:
            node.returns = _type_expression(node.returns)
        self.generic_visit(node)
        return node

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self._returning(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return self._returning(node)

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        node.annotation = _type_expression(node.annotation)
        self.generic_visit(node)
        return node

    @override
    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if ast.unparse(node.func) in CASTS and node.args:
            node.args[0] = _type_expression(node.args[0])
        return node


def unquoted(tree: ast.Module) -> ast.Module:
    return ast.fix_missing_locations(cast("ast.Module", _Unquoted().visit(copy.deepcopy(tree))))


def _resolved(tree: ast.Module, names: Mapping[str, str]) -> ast.Module:
    return ast.fix_missing_locations(cast("ast.Module", _Resolved(names).visit(copy.deepcopy(tree))))


def imports_resolved(tree: ast.Module) -> ast.Module:
    return _resolved(tree, aliases(tree))


def canonical(tree: ast.Module) -> ast.Module:
    return _resolved(tree, {**bindings(tree), **aliases(tree)})
