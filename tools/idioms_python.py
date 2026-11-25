import ast
from collections.abc import Iterator

from tools.findings import Finding

CONTAINERS = frozenset({"OrderedDict", "Counter", "ChainMap"})
GENERIC_CALLED = (
    "a generic class called with its type arguments, which Python 3.12.0 fails on for a frozen slotted dataclass, "
    "as typing sets __orig_class__ on what it made: call the class alone and annotate what receives it"
)
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
TLS_CONTEXT = "ssl.SSLContext"
TLS_FLOORS = frozenset({"ssl.TLSVersion.TLSv1_2", "ssl.TLSVersion.TLSv1_3"})
NO_TLS_FLOOR = (
    "a TLS context made without the oldest version it takes, which a scanner reads as TLS 1.0 allowed: "
    "set its minimum_version = ssl.TLSVersion.TLSv1_2, or TLSv1_3, in the function that makes it"
)


def _class_named(node: ast.expr) -> bool:
    name = ast.unparse(node).rsplit(".", 1)[-1]
    return name.lstrip("_")[:1].isupper() and not name.isupper() and name not in CONTAINERS


def generics_called(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript) and _class_named(node.func.value):
            yield Finding(path, node.lineno, GENERIC_CALLED)


def _placed(node: ast.AST, scope: ast.AST) -> Iterator[tuple[ast.AST, ast.AST]]:
    for child in ast.iter_child_nodes(node):
        inner = child if isinstance(child, FUNCTIONS) else scope
        yield inner, child
        yield from _placed(child, inner)


def _floors_tls(node: ast.AST) -> bool:
    match node:
        case ast.Assign(targets=[ast.Attribute(attr="minimum_version")], value=floor):
            return ast.unparse(floor) in TLS_FLOORS
        case _:
            return False


def tls_without_a_floor(path: str, tree: ast.Module) -> Iterator[Finding]:
    placed = list(_placed(tree, tree))
    floored = {id(scope) for scope, node in placed if _floors_tls(node)}
    for scope, node in placed:
        if isinstance(node, ast.Call) and ast.unparse(node.func) == TLS_CONTEXT and id(scope) not in floored:
            yield Finding(path, node.lineno, NO_TLS_FLOOR)
