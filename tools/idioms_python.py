import ast
from collections.abc import Iterator

from tools.findings import Finding

CONTAINERS = frozenset({"OrderedDict", "Counter", "ChainMap"})
GENERIC_CALLED = (
    "a generic class called with its type arguments, which Python 3.12.0 fails on for a frozen slotted dataclass, "
    "as typing sets __orig_class__ on what it made: call the class alone and annotate what receives it"
)


def _class_named(node: ast.expr) -> bool:
    name = ast.unparse(node).rsplit(".", 1)[-1]
    return name.lstrip("_")[:1].isupper() and not name.isupper() and name not in CONTAINERS


def generics_called(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript) and _class_named(node.func.value):
            yield Finding(path, node.lineno, GENERIC_CALLED)
