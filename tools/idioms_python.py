import ast
from collections.abc import Iterator

from tools.findings import Finding

CONTAINERS = frozenset({"OrderedDict", "Counter", "ChainMap"})
GENERIC_CALLED = (
    "a generic class called with its type arguments, which Python 3.12.0 fails on for a frozen slotted dataclass, "
    "as typing sets __orig_class__ on what it made: call the class alone and annotate what receives it"
)
ESCAPERS = frozenset({"labelled", "printable", "printable_text"})
PAINTERS = frozenset({"paint"})
PAINTED_ESCAPED = (
    "a painted text handed to labelled() or printable(), which escape its colour's control characters, so the "
    "terminal shows them as text (\\x1b[2m...): escape the rest first, then add the painted part"
)
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
TLS_CONTEXT = "ssl.SSLContext"
TLS_FLOORS = frozenset({"ssl.TLSVersion.TLSv1_2", "ssl.TLSVersion.TLSv1_3"})
NO_TLS_FLOOR = (
    "a TLS context made without the oldest version it takes, which a scanner reads as TLS 1.0 allowed: "
    "bind it to a name and set that name's minimum_version = ssl.TLSVersion.TLSv1_2, or TLSv1_3, "
    "in the function that makes it"
)


def _class_named(node: ast.expr) -> bool:
    name = ast.unparse(node).rsplit(".", 1)[-1]
    return name.lstrip("_")[:1].isupper() and not name.isupper() and name not in CONTAINERS


def generics_called(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript) and _class_named(node.func.value):
            yield Finding(path, node.lineno, GENERIC_CALLED)


def _last_name(call: ast.Call) -> str:
    return ast.unparse(call.func).rsplit(".", 1)[-1]


def _painted(node: ast.AST) -> bool:
    return any(isinstance(inner, ast.Call) and _last_name(inner) in PAINTERS for inner in ast.walk(node))


def painted_escaped(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _last_name(node) in ESCAPERS:
            given = [*node.args, *(keyword.value for keyword in node.keywords)]
            if any(map(_painted, given)):
                yield Finding(path, node.lineno, PAINTED_ESCAPED)


def _placed(node: ast.AST, scope: ast.AST) -> Iterator[tuple[ast.AST, ast.AST]]:
    for child in ast.iter_child_nodes(node):
        inner = child if isinstance(child, FUNCTIONS) else scope
        yield inner, child
        yield from _placed(child, inner)


def _floored(node: ast.AST) -> str | None:
    match node:
        case ast.Assign(targets=[ast.Attribute(attr="minimum_version", value=held)], value=floor):
            return ast.unparse(held) if ast.unparse(floor) in TLS_FLOORS else None
        case _:
            return None


def _named(node: ast.AST) -> tuple[int, str] | None:
    match node:
        case (
            ast.Assign(targets=[target], value=ast.Call() as made)
            | ast.AnnAssign(target=target, value=ast.Call() as made)
        ):
            return id(made), ast.unparse(target)
        case _:
            return None


def tls_without_a_floor(path: str, tree: ast.Module) -> Iterator[Finding]:
    placed = list(_placed(tree, tree))
    floored = {(id(scope), held) for scope, node in placed if (held := _floored(node)) is not None}
    names = dict(named for _, node in placed if (named := _named(node)) is not None)
    for scope, node in placed:
        held = (id(scope), names.get(id(node)))
        if isinstance(node, ast.Call) and ast.unparse(node.func) == TLS_CONTEXT and held not in floored:
            yield Finding(path, node.lineno, NO_TLS_FLOOR)
