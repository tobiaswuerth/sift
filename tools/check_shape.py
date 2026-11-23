import ast
from collections.abc import Iterator

from tools.canonical import aliases
from tools.findings import PYTHON_ROOTS, Finding, found, report

MOST_LINES = 400
MOST_METHODS = 10
CORE = ("src/sift/api/", "src/sift/domain/")
SIFT = "sift."
VALUE_BASES = frozenset({"NamedTuple", "Enum", "StrEnum", "IntEnum", "Flag", "IntFlag", "Protocol"})
ERROR_SUFFIXES = ("Error", "Exception")
ARRAY_TYPES = frozenset({"NDArray", "Ids", "Mask", "Scores", "Floats", "Halves", "Bits", "Ascending", "Coded"})
MUTABLE_FACTORIES = frozenset(
    {"list", "dict", "set", "bytearray", "defaultdict", "OrderedDict", "deque", "Counter", "ChainMap"}
)
IMMUTABLE_TYPES = frozenset(
    {"frozenset", "tuple", "MappingProxyType", "str", "int", "float", "bytes", "bool", "range", "Path", "PurePosixPath"}
)
IMMUTABLE_FACTORIES = IMMUTABLE_TYPES | frozenset(
    {"compile", "Struct", "NewType", "TypeVar", "dtype", "hexdigest", "read_only"}
)
REMEDY = (
    "Split what grew too large by responsibility. Hold module values in immutable types (tuple, frozenset, "
    "MappingProxyType, an array made read_only), or make them in a function whose return type is one. "
    "In sift.api and sift.domain a class is a frozen slotted dataclass, a NamedTuple, an Enum, a Protocol or an error."
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


def _returning_immutables(tree: ast.Module) -> frozenset[str]:
    functions = (
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.returns is not None and _name(node.returns) in IMMUTABLE_TYPES
    )
    values = (
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and (_is_frozen_slotted_dataclass(node) or any(map(_is_value_base, node.bases)))
    )
    imported = (name for name, whole in aliases(tree).items() if whole.startswith(SIFT) and name[:1].isupper())
    return frozenset((*functions, *values, *imported))


def _made_immutable(call: ast.Call, trusted: frozenset[str]) -> bool:
    name = _name(call.func)
    return name in IMMUTABLE_FACTORIES or (isinstance(call.func, ast.Name) and name in trusted)


def _immutable(value: ast.expr, trusted: frozenset[str]) -> bool:
    match value:
        case ast.Constant() | ast.Name() | ast.Attribute() | ast.Lambda() | ast.JoinedStr() | ast.Subscript():
            return True
        case ast.Tuple(elts=items):
            return all(_immutable(item, trusted) for item in items)
        case ast.Starred(value=inner) | ast.UnaryOp(operand=inner):
            return _immutable(inner, trusted)
        case ast.BinOp(left=left, right=right) | ast.IfExp(body=left, orelse=right):
            return _immutable(left, trusted) and _immutable(right, trusted)
        case ast.Call():
            return _made_immutable(value, trusted)
        case _:
            return False


def mutable_module_values(path: str, tree: ast.Module) -> Iterator[Finding]:
    trusted = _returning_immutables(tree)
    for node in tree.body:
        value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
        if value is not None and not _immutable(value, trusted):
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
