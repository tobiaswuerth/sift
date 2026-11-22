import ast
import re
import unicodedata
from collections.abc import Iterator
from types import MappingProxyType

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
EXITS = frozenset({"SystemExit", "exit", "quit"})
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
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in IO_BUILTINS:
            yield Finding(path, node.lineno, f"{node.func.id}() in the core: return the value and let the caller act")


def exits_outside_interfaces(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in EXITS:
            yield Finding(path, node.lineno, f"{node.id} outside sift.interfaces: raise a SiftError")


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


def idioms(path: str, source: str) -> tuple[Finding, ...]:
    tree = ast.parse(source)
    in_tests, in_sift = path.startswith("tests/"), path.startswith("src/sift/")
    return (
        *invisible_characters(path, source),
        *numbered_placeholders(path, tree),
        *(() if in_tests else banned_methods(path, tree)),
        *(io_in_the_core(path, tree) if path.startswith(CORE) else ()),
        *(exits_outside_interfaces(path, tree) if in_sift and not path.startswith(INTERFACES) else ()),
        *(environment_outside_settings(path, tree) if in_sift and path != SETTINGS else ()),
        *(patching(path, tree) if in_tests else ()),
        *(clocks(path, tree) if in_tests else ()),
        *(databases_left_open(path, tree) if in_tests else ()),
    )


def main(_argv: list[str]) -> int:
    return report(found(idioms, PYTHON_ROOTS), REMEDY)
