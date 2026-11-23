import ast
import copy
from collections import Counter
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


def _resolved(tree: ast.Module, names: Mapping[str, str]) -> ast.Module:
    return ast.fix_missing_locations(cast("ast.Module", _Resolved(names).visit(copy.deepcopy(tree))))


def imports_resolved(tree: ast.Module) -> ast.Module:
    return _resolved(tree, aliases(tree))


def canonical(tree: ast.Module) -> ast.Module:
    return _resolved(tree, {**bindings(tree), **aliases(tree)})
