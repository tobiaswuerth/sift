import ast
from collections.abc import Iterator

from tools.findings import PYTHON_ROOTS, Finding, found, report

MOST_LINES = 400
MOST_METHODS = 10
CORE = ("src/sift/api/", "src/sift/domain/")
VALUE_BASES = frozenset({"NamedTuple", "Enum", "StrEnum", "IntEnum", "Flag", "IntFlag", "Protocol"})
ERROR_SUFFIXES = ("Error", "Exception")
ARRAY_TYPES = frozenset({"NDArray", "Ids", "Mask", "Scores", "Floats", "Halves", "Bits", "Ascending", "Coded"})
MUTABLE_FACTORIES = frozenset(
    {"list", "dict", "set", "bytearray", "defaultdict", "OrderedDict", "deque", "Counter", "ChainMap"}
)
MUTABLE_DISPLAYS = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)
REMEDY = (
    "Split what grew too large by responsibility. Hold module values in immutable types (tuple, frozenset, "
    "MappingProxyType). In sift.api and sift.domain a class is a frozen slotted dataclass, a NamedTuple, an Enum, "
    "a Protocol or an error."
)


def too_long(path: str, source: str) -> Iterator[Finding]:
    lines = source.count("\n")
    if lines > MOST_LINES:
        yield Finding(path, 1, f"module of {lines} lines, more than {MOST_LINES}")


def too_many_methods(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = sum(isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) for item in node.body)
            if methods > MOST_METHODS:
                yield Finding(path, node.lineno, f"class {node.name} has {methods} methods, more than {MOST_METHODS}")


def _name(node: ast.expr) -> str:
    match node:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
        case ast.Subscript(value=inner):
            return _name(inner)
        case _:
            return ""


def _is_mutable(value: ast.expr) -> bool:
    return isinstance(value, MUTABLE_DISPLAYS) or (
        isinstance(value, ast.Call) and _name(value.func) in MUTABLE_FACTORIES
    )


def mutable_module_values(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in tree.body:
        value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
        if value is not None and _is_mutable(value):
            yield Finding(path, node.lineno, "mutable value at module level")


def _is_true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_frozen_slotted_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call) and _name(decorator.func) == "dataclass":
            options = {keyword.arg: keyword.value for keyword in decorator.keywords}
            if _is_true(options.get("frozen")) and _is_true(options.get("slots")):
                return True
    return False


def _is_value_base(base: ast.expr) -> bool:
    name = _name(base)
    return name in VALUE_BASES or name.endswith(ERROR_SUFFIXES)


def classes_that_are_no_values(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and not (
            _is_frozen_slotted_dataclass(node) or any(map(_is_value_base, node.bases))
        ):
            yield Finding(path, node.lineno, f"class {node.name} is no frozen slotted dataclass or value type")


def _names_in(node: ast.expr) -> set[str]:
    return {each.id for each in ast.walk(node) if isinstance(each, ast.Name)}


def _compares(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call) and _name(decorator.func) == "dataclass":
            options = {keyword.arg: keyword.value for keyword in decorator.keywords}
            found = options.get("eq")
            return not (isinstance(found, ast.Constant) and found.value is False)
    return False


def arrays_compared(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _compares(node):
            held = {
                name for item in node.body if isinstance(item, ast.AnnAssign) for name in _names_in(item.annotation)
            }
            if held & ARRAY_TYPES:
                yield Finding(path, node.lineno, f"class {node.name} holds arrays and compares them: eq=False")


def shape(path: str, source: str) -> tuple[Finding, ...]:
    tree = ast.parse(source)
    return (
        *too_long(path, source),
        *too_many_methods(path, tree),
        *(arrays_compared(path, tree) if path.startswith("src/") else ()),
        *(() if path.startswith("tests/") else mutable_module_values(path, tree)),
        *(classes_that_are_no_values(path, tree) if path.startswith(CORE) else ()),
    )


def main(_argv: list[str]) -> int:
    return report(found(shape, PYTHON_ROOTS), REMEDY)
