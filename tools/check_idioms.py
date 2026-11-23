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
STORE_OPENED_BY_HAND = (
    "a store opened by hand outside sift.storage: open it with connected() or read_store(), "
    "which close it and say why a store cannot be used"
)
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
TEXT_CUT = re.compile(r"substr\(\s*(?:\w+\.)?text\b", re.IGNORECASE)
TEXT_CUT_IN_SQL = (
    "a document's text cut in SQL, which walks it from its start for every chunk: "
    "read the text once and cut its spans in Python"
)
FLAG = re.compile(r"(?<![\w-])(?:--[a-z][a-z-]*|-[kcf])(?![\w-])")
INSTALLER = re.compile(r"\buv [a-z]")
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


def _owned(target: ast.expr) -> list[ast.expr]:
    match target:
        case ast.Tuple(elts=items) | ast.List(elts=items) | ast.Starred(value=ast.Tuple(elts=items)):
            return [owned for item in items for owned in _owned(item)]
        case ast.Starred(value=inner):
            return _owned(inner)
        case (
            ast.Subscript(value=ast.Call(func=ast.Name(id="vars"), args=[owner]))
            | ast.Subscript(value=ast.Attribute(attr="__dict__", value=owner))
        ):
            return [ast.Attribute(owner, "item", ast.Store())]
        case _:
            return [target]


def _changed(node: ast.AST) -> list[ast.expr]:
    match node:
        case ast.Assign(targets=targets) | ast.Delete(targets=targets):
            return [owned for target in targets for owned in _owned(target)]
        case ast.AugAssign(target=target) | ast.AnnAssign(target=target):
            return _owned(target)
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


def _named_tuples(tree: ast.Module) -> dict[str, list[ast.expr]]:
    found: dict[str, list[ast.expr]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            found.update(dict.fromkeys(names, node.value.elts))
        elif (
            isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Tuple) and isinstance(node.target, ast.Name)
        ):
            found[node.target.id] = node.value.elts
    return found


def _caught(node: ast.expr, named: dict[str, list[ast.expr]], seen: frozenset[str]) -> Iterator[str]:
    if isinstance(node, ast.Tuple):
        for each in node.elts:
            yield from _caught(each, named, seen)
    elif isinstance(node, ast.Name) and node.id in named:
        for each in () if node.id in seen else named[node.id]:
            yield from _caught(each, named, seen | {node.id})
    else:
        yield ast.unparse(node).split(".")[-1]


def recursion_caught(path: str, tree: ast.Module) -> Iterator[Finding]:
    named = _named_tuples(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ExceptHandler)
            and node.type is not None
            and "RecursionError" in _caught(node.type, named, frozenset())
        ):
            yield Finding(path, node.lineno, RECURSION_CAUGHT)


def _flagged(text: str) -> bool:
    return bool(FLAG.search(text)) and not INSTALLER.search(text)


def flags_outside_interfaces(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _flagged(node.value):
            yield Finding(path, node.lineno, FLAG_OUTSIDE)


def text_cut_in_sql(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and TEXT_CUT.search(node.value):
            yield Finding(path, node.lineno, TEXT_CUT_IN_SQL)


def handles_made(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func).rsplit(".", 1)[-1] in HANDLES:
            yield Finding(path, node.lineno, HANDLE_MADE)


def stores_opened_by_hand(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func).rsplit(".", 1)[-1] == "open_store":
            yield Finding(path, node.lineno, STORE_OPENED_BY_HAND)


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
        *(stores_opened_by_hand(path, tree) if in_sift and not path.startswith(STORAGE) else ()),
        *(text_cut_in_sql(path, tree) if in_sift else ()),
        *(patching(path, tree) if in_tests else ()),
        *(sift_patched(path, tree) if in_tests else ()),
        *(clocks(path, tree) if in_tests else ()),
        *(databases_left_open(path, tree) if in_tests else ()),
    )


def main(_argv: list[str]) -> int:
    return report(found(idioms, PYTHON_ROOTS), REMEDY)
