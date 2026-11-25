import ast
import re
from collections.abc import Iterator

from tools.findings import Finding

SIFT = "src/sift/"
TESTS = "tests/"
STORAGE = "src/sift/storage/"
NUMBERED_PLACEHOLDER = re.compile(r"\?[0-9]")
HANDLE_MAKERS = frozenset({"src/sift/storage/database.py", "src/sift/storage/keywords.py"})
HANDLES = frozenset({"Writing", "WritingDocuments"})
HANDLE_MADE = "a write handle made by hand: open one with writing() or writing_documents()"
STORE_OPENED_BY_HAND = (
    "a store opened by hand outside sift.storage: open it with connected() or read_store(), "
    "which close it and say why a store cannot be used"
)
SQL_CALLS = frozenset({"execute", "executemany", "executescript"})
RAW_SQL = "SQL run on a transaction's connection outside sift.storage: add a storage function and call it"
LAYOUT_FROM_ITS_END = (
    "a layout cut from its end, which the next migration moves: "
    "cut it before the migration it means, found by what that migration holds"
)
NUMBERED = (
    "a numbered SQL placeholder, which Python 3.12.0 to 3.12.3 take for a named one: "
    "bind by name from a dict, or give each plain ? its own value"
)
LEFT_OPEN = (
    "sqlite3.connect left to the garbage collector, a ResourceWarning in whichever test runs then: "
    "open it in `with closing(sqlite3.connect(...)) as db:`"
)
TEXT_CUT = re.compile(r"substr(?:ing)?\(\s*(?:\w+\.)?text\b", re.IGNORECASE)
TEXT_CUT_IN_SQL = (
    "a document's text cut in SQL, which walks it from its start for every chunk: "
    "read the text once and cut its spans in Python"
)


def _texts(tree: ast.Module) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def numbered_placeholders(path: str, tree: ast.Module) -> Iterator[Finding]:
    for line, text in _texts(tree):
        if NUMBERED_PLACEHOLDER.search(text):
            yield Finding(path, line, NUMBERED)


def text_cut_in_sql(path: str, tree: ast.Module) -> Iterator[Finding]:
    for line, text in _texts(tree):
        if TEXT_CUT.search(text):
            yield Finding(path, line, TEXT_CUT_IN_SQL)


def databases_left_open(path: str, tree: ast.Module) -> Iterator[Finding]:
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    closed = {
        id(argument)
        for call in calls
        if ast.unparse(call.func) in ("closing", "contextlib.closing")
        for argument in call.args
    }
    for call in calls:
        if ast.unparse(call.func) == "sqlite3.connect" and id(call) not in closed:
            yield Finding(path, call.lineno, LEFT_OPEN)


def _called(tree: ast.Module, last: frozenset[str]) -> Iterator[ast.Call]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func).rsplit(".", 1)[-1] in last:
            yield node


def handles_made(path: str, tree: ast.Module) -> Iterator[Finding]:
    yield from (Finding(path, node.lineno, HANDLE_MADE) for node in _called(tree, HANDLES))


def stores_opened_by_hand(path: str, tree: ast.Module) -> Iterator[Finding]:
    yield from (Finding(path, node.lineno, STORE_OPENED_BY_HAND) for node in _called(tree, frozenset({"open_store"})))


def raw_sql(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        match node:
            case ast.Attribute(attr=called, value=ast.Attribute(attr="connection")) if called in SQL_CALLS:
                yield Finding(path, node.lineno, RAW_SQL)
            case _:
                pass


def _from_the_end(bound: ast.expr | None) -> bool:
    return isinstance(bound, ast.UnaryOp) and isinstance(bound.op, ast.USub)


def layouts_cut_from_their_end(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and ast.unparse(node.value).endswith("LAYOUT"):
            cut = node.slice
            bounds = (cut.lower, cut.upper) if isinstance(cut, ast.Slice) else (cut,)
            if any(map(_from_the_end, bounds)):
                yield Finding(path, node.lineno, LAYOUT_FROM_ITS_END)


def storage_idioms(path: str, tree: ast.Module) -> tuple[Finding, ...]:
    in_sift, outside_storage = path.startswith(SIFT), not path.startswith(STORAGE)
    return (
        *numbered_placeholders(path, tree),
        *(handles_made(path, tree) if in_sift and path not in HANDLE_MAKERS else ()),
        *(raw_sql(path, tree) if in_sift and outside_storage else ()),
        *(stores_opened_by_hand(path, tree) if in_sift and outside_storage else ()),
        *(text_cut_in_sql(path, tree) if in_sift else ()),
        *(databases_left_open(path, tree) if path.startswith(TESTS) else ()),
        *layouts_cut_from_their_end(path, tree),
    )
