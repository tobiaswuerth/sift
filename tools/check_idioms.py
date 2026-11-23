import ast
import re
import unicodedata
from collections.abc import Iterator
from types import MappingProxyType

from tools.canonical import canonical
from tools.findings import PYTHON_ROOTS, Finding, found, report

CORE = ("src/sift/api/", "src/sift/domain/")
INTERFACES = "src/sift/interfaces/"
DIGITS = (
    'match digits as re.fullmatch("[0-9]+", text): this also takes superscripts and other scripts, which int() refuses'
)
BANNED_METHODS = MappingProxyType(
    {
        "splitlines": 'split on "\\n": splitlines also ends a line at U+2028, U+2029, NEL and five control characters',
        "isdigit": DIGITS,
        "isdecimal": DIGITS,
        "isnumeric": DIGITS,
    }
)
UNSEEN_CATEGORIES = frozenset({"Cc", "Cf", "Cn", "Co", "Cs", "Me", "Mn", "Zl", "Zp", "Zs"})
SEEN = frozenset({" "})
LOOKS_LIKE_DAMAGE = frozenset({"\N{REPLACEMENT CHARACTER}"})
NUMBERED_PLACEHOLDER = re.compile(r"\?[0-9]")
IO_BUILTINS = frozenset({"open", "input", "print"})
IO_CALLS = IO_BUILTINS | frozenset(f"builtins.{name}" for name in IO_BUILTINS)
EXITS = frozenset({"SystemExit", "exit", "quit", "sys.exit", "builtins.exit", "builtins.quit", "builtins.SystemExit"})
NOW = frozenset({"datetime.datetime.now", "datetime.datetime.today", "datetime.datetime.utcnow", "datetime.date.today"})
CLOCK_READ = "the clock read outside sift.interfaces: take the time as a value, and let the interface read the clock"
PATCHED = "a sift module or class changed by a test: pass a fake, a value or an environment in instead"
SETTERS = frozenset({"setattr", "delattr", "builtins.setattr", "builtins.delattr"})
HANDLE_MAKERS = frozenset({"src/sift/storage/database.py", "src/sift/storage/keywords.py"})
HANDLES = frozenset({"Writing", "WritingDocuments"})
HANDLE_MADE = "a write handle made by hand: open one with writing() or writing_documents()"
STORAGE = "src/sift/storage/"
SQL_CALLS = frozenset({"execute", "executemany", "executescript"})
RAW_SQL = "SQL run on a transaction's connection outside sift.storage: add a storage function and call it"
NUMBERED = (
    "a numbered SQL placeholder, which Python 3.12.0 to 3.12.3 take for a named one: "
    "bind by name from a dict, or give each plain ? its own value"
)
LEFT_OPEN = (
    "sqlite3.connect left to the garbage collector, a ResourceWarning in whichever test runs then: "
    "open it in `with closing(sqlite3.connect(...)) as db:`"
)
SETTINGS = "src/sift/settings.py"
ENVIRONMENT = frozenset({"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"})
ENVIRONMENT_READ = "the environment read outside sift.settings: parse it there and pass the Settings in"
CLOCKS = frozenset({"time", "timeit"})
CLOCK_IN_A_TEST = (
    "a clock in a test, which passes or fails by the machine and its load: count what costs the time "
    "(calls, characters looked at, rows read), or pass a clock in"
)
RECURSION_CAUGHT = (
    "RecursionError caught: the interpreter's depth limit decides, and the work done before it is lost; "
    "bound the depth yourself before descending, and refuse what is deeper"
)
FLAG = re.compile(r"(?<![\w-])(?:--[a-z][a-z-]*|-[kcf])(?![\w-])")
COMMAND = re.compile(r"\b(?:sift|uv) [a-z]")
FLAG_OUTSIDE = "CLI syntax outside sift.interfaces: say what to do, and let the interface name its flag"
REMEDY = "Each finding names its fix."


def _unseen(character: str) -> bool:
    return character in LOOKS_LIKE_DAMAGE or (
        character not in SEEN and unicodedata.category(character) in UNSEEN_CATEGORIES
    )


def invisible_characters(path: str, source: str) -> Iterator[Finding]:
    for number, line in enumerate(source.split("\n"), 1):
        for character in sorted(filter(_unseen, set(line))):
            yield Finding(path, number, f"U+{ord(character):04X} written as itself: write it as an escape")


def banned_methods(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in BANNED_METHODS:
            yield Finding(path, node.lineno, f".{node.attr}: {BANNED_METHODS[node.attr]}")


def numbered_placeholders(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and NUMBERED_PLACEHOLDER.search(node.value):
            yield Finding(path, node.lineno, NUMBERED)


def io_in_the_core(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (called := ast.unparse(node.func)) in IO_CALLS:
            yield Finding(path, node.lineno, f"{called}() in the core: return the value and let the caller act")


def exits_outside_interfaces(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name | ast.Attribute) and (named := ast.unparse(node)) in EXITS:
            yield Finding(path, node.lineno, f"{named} outside sift.interfaces: raise a SiftError")


def clock_read(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and ast.unparse(node) in NOW:
            yield Finding(path, node.lineno, CLOCK_READ)


def _of_sift(node: ast.expr) -> bool:
    return isinstance(node, ast.Attribute) and ast.unparse(node.value).split(".")[0] == "sift"


def _changed(node: ast.AST) -> list[ast.expr]:
    match node:
        case ast.Assign(targets=targets) | ast.Delete(targets=targets):
            return targets
        case ast.AugAssign(target=target) | ast.AnnAssign(target=target):
            return [target]
        case ast.Call(func=func, args=[owner, ast.Constant(value=str(name)), *_]) if ast.unparse(func) in SETTERS:
            return [ast.Attribute(owner, name, ast.Store())]
        case _:
            return []


def sift_patched(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if any(map(_of_sift, _changed(node))):
            yield Finding(path, getattr(node, "lineno", 1), PATCHED)


def patching(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if (isinstance(node, ast.arg) and node.arg == "monkeypatch") or (
            isinstance(node, ast.Attribute) and node.attr == "MonkeyPatch"
        ):
            yield Finding(path, node.lineno, "monkeypatch: pass a fake, a value or an environment in instead")


def environment_outside_settings(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and ast.unparse(node.value) == "os" and node.attr in ENVIRONMENT) or (
            isinstance(node, ast.ImportFrom) and node.module == "os" and ENVIRONMENT & {a.name for a in node.names}
        ):
            yield Finding(path, node.lineno, ENVIRONMENT_READ)


def _imported(node: ast.Import | ast.ImportFrom) -> list[str]:
    return [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]


def clocks(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom) and CLOCKS & {n.split(".")[0] for n in _imported(node)}:
            yield Finding(path, node.lineno, CLOCK_IN_A_TEST)


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


def recursion_caught(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            caught = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            if any(ast.unparse(each).split(".")[-1] == "RecursionError" for each in caught):
                yield Finding(path, node.lineno, RECURSION_CAUGHT)


def _flagged(text: str) -> bool:
    return bool(FLAG.search(text)) and not COMMAND.search(text)


def flags_outside_interfaces(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _flagged(node.value):
            yield Finding(path, node.lineno, FLAG_OUTSIDE)


def handles_made(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func).rsplit(".", 1)[-1] in HANDLES:
            yield Finding(path, node.lineno, HANDLE_MADE)


def raw_sql(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        match node:
            case ast.Attribute(attr=called, value=ast.Attribute(attr="connection")) if called in SQL_CALLS:
                yield Finding(path, node.lineno, RAW_SQL)
            case _:
                pass


def idioms(path: str, source: str) -> tuple[Finding, ...]:
    tree = canonical(ast.parse(source))
    in_tests, in_sift = path.startswith("tests/"), path.startswith("src/sift/")
    outside_interfaces = in_sift and not path.startswith(INTERFACES)
    return (
        *invisible_characters(path, source),
        *numbered_placeholders(path, tree),
        *(() if in_tests else banned_methods(path, tree)),
        *(() if in_tests else recursion_caught(path, tree)),
        *(io_in_the_core(path, tree) if path.startswith(CORE) else ()),
        *(exits_outside_interfaces(path, tree) if outside_interfaces else ()),
        *(flags_outside_interfaces(path, tree) if outside_interfaces else ()),
        *(clock_read(path, tree) if outside_interfaces or in_tests else ()),
        *(environment_outside_settings(path, tree) if in_sift and path != SETTINGS else ()),
        *(handles_made(path, tree) if in_sift and path not in HANDLE_MAKERS else ()),
        *(raw_sql(path, tree) if in_sift and not path.startswith(STORAGE) else ()),
        *(patching(path, tree) if in_tests else ()),
        *(sift_patched(path, tree) if in_tests else ()),
        *(clocks(path, tree) if in_tests else ()),
        *(databases_left_open(path, tree) if in_tests else ()),
    )


def main(_argv: list[str]) -> int:
    return report(found(idioms, PYTHON_ROOTS), REMEDY)
