import ast
import copy
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast, final, override

ATTRIBUTE_GETTERS = frozenset({"getattr", "builtins.getattr"})


def aliases(tree: ast.Module) -> Mapping[str, str]:
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update((alias.asname, alias.name) for alias in node.names if alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.update((alias.asname or alias.name, f"{node.module}.{alias.name}") for alias in node.names)
    return MappingProxyType(found)


@final
class _Resolved(ast.NodeTransformer):
    def __init__(self, names: Mapping[str, str]) -> None:
        self.names = names

    @override
    def visit_Name(self, node: ast.Name) -> ast.expr:
        dotted = self.names.get(node.id)
        if dotted is None or not isinstance(node.ctx, ast.Load):
            return node
        return ast.copy_location(ast.parse(dotted, mode="eval").body, node)

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


def canonical(tree: ast.Module) -> ast.Module:
    return ast.fix_missing_locations(cast("ast.Module", _Resolved(aliases(tree)).visit(copy.deepcopy(tree))))
