import ast
import re
import unicodedata
from collections.abc import Callable, Iterator, Mapping
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import TypeGuard

from tools.canonical import canonical
from tools.findings import PYTHON_ROOTS, Finding, found, report
from tools.idioms_storage import storage_idioms

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
IO_BUILTINS = frozenset({"open", "input", "print"})
IO_CALLS = IO_BUILTINS | frozenset(f"builtins.{name}" for name in IO_BUILTINS)
EXITS = frozenset({"SystemExit", "exit", "quit", "sys.exit", "builtins.exit", "builtins.quit", "builtins.SystemExit"})
HARD_EXITS = frozenset({"os._exit", "os.abort"})
ENDS_ITS_OWN_PROCESS = frozenset({"src/sift/ingest/sandbox.py"})
NOW = frozenset({"datetime.datetime.now", "datetime.datetime.today", "datetime.datetime.utcnow", "datetime.date.today"})
TIMERS = frozenset(
    f"time.{name}{unit}"
    for name in ("time", "perf_counter", "monotonic", "process_time", "thread_time")
    for unit in ("", "_ns")
)
CLOCK_READ = "the clock read outside sift.interfaces: take the time as a value, and let the interface read the clock"
PATCHED = "a sift module or class changed by a test: pass a fake, a value or an environment in instead"
SETTERS = frozenset({"setattr", "delattr", "builtins.setattr", "builtins.delattr"})
MODULE_TABLE = "sys.modules"
MODULE_LOADERS = frozenset({"import_module", "importlib.import_module"})
SIFT = "sift"
SETTINGS = "src/sift/settings.py"
ENVIRONMENT = frozenset({"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"})
ENVIRONMENT_MODULES = frozenset({"os", "nt", "posix"})
NAMESPACES = frozenset({"vars", "builtins.vars"})
ENVIRONMENT_READ = "the environment read outside sift.settings: parse it there and pass the Settings in"
CLOCKS = frozenset({"time", "timeit"})
EXAMPLE_COUNTS = frozenset({"tests/examples.py", "tests/conftest.py"})
PINNED_EXAMPLES = (
    "an example count pinned in a test, which the explore session cannot widen: "
    "use few(n) or more(n) from tests/examples.py"
)
WAITS = "tests/waits.py"
WAIT_DROPPED = (
    "a wait or join whose outcome is dropped, so a test that should fail hangs or goes on: "
    "use joined, waited, crossed, until or paused from tests/waits.py"
)
CLOCK_IN_A_TEST = (
    "a clock in a test, which passes or fails by the machine and its load: count what costs the time "
    "(calls, characters looked at, rows read), or pass a clock in"
)
FIRST_PARTY = "src"
RECURSION_CAUGHT = (
    "RecursionError caught, by name or as a RuntimeError, Exception or BaseException: the interpreter's depth limit "
    "decides, and the work done before it is lost; bound the depth yourself before descending, and let RecursionError "
    "through: re-raise what is caught, or put `except RecursionError: raise` first"
)
RECURSION_AND_BASES = frozenset({"RecursionError", "RuntimeError", "Exception", "BaseException"})
PATTERN_CALLS = frozenset(
    f"re.{name}" for name in ("compile", "match", "search", "fullmatch", "sub", "subn", "split", "findall", "finditer")
)
DIGIT_CLASS = re.compile(r"(?<!\\)(?:\\\\)*\\[dD]")
UNICODE_DIGITS = "\\d in a pattern, which also matches the digits of other scripts that int() refuses: write [0-9]"
FLAG = re.compile(r"(?<![\w-])(?:--[a-z][a-z-]*|-[kcf])(?![\w-])")
INSTALLER = re.compile(r"\buv [a-z]")
COMMANDS = (
    "activate",
    "add",
    "bench",
    "deactivate",
    "embed",
    "get",
    "index",
    "jobs",
    "links",
    "mentions",
    "move",
    "query",
    "rm",
    "search",
    "serve",
    "similar",
    "status",
    "ui",
    "unresolved",
    "update",
    "vsearch",
    "worker",
)
COMMAND = re.compile(rf"\bsift (?:{'|'.join(COMMANDS)})\b")
FLAG_OUTSIDE = "CLI syntax outside sift.interfaces: say what to do, and let the interface name its command or flag"
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


def io_in_the_core(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (called := ast.unparse(node.func)) in IO_CALLS:
            yield Finding(path, node.lineno, f"{called}() in the core: return the value and let the caller act")


def exits_outside_interfaces(path: str, tree: ast.Module) -> Iterator[Finding]:
    ended = EXITS if path in ENDS_ITS_OWN_PROCESS else EXITS | HARD_EXITS
    for node in ast.walk(tree):
        if isinstance(node, ast.Name | ast.Attribute) and (named := ast.unparse(node)) in ended:
            yield Finding(path, node.lineno, f"{named} outside sift.interfaces: raise a SiftError")


def _reads_the_clock(node: ast.AST) -> TypeGuard[ast.Attribute | ast.Call]:
    return (isinstance(node, ast.Attribute) and ast.unparse(node) in NOW) or (
        isinstance(node, ast.Call) and ast.unparse(node.func) in TIMERS
    )


def clock_read(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if _reads_the_clock(node):
            yield Finding(path, node.lineno, CLOCK_READ)


def _sift_module(node: ast.expr) -> bool:
    match node:
        case ast.Subscript(value=table, slice=ast.Constant(value=str(name))) if ast.unparse(table) == MODULE_TABLE:
            return name.split(".")[0] == SIFT
        case ast.Call(func=loader, args=[ast.Constant(value=str(name)), *_]) if ast.unparse(loader) in MODULE_LOADERS:
            return name.split(".")[0] == SIFT
        case _:
            return False


def _of_sift(node: ast.expr) -> bool:
    return _sift_module(node) or (
        isinstance(node, ast.Attribute) and (ast.unparse(node.value).split(".")[0] == SIFT or _sift_module(node.value))
    )


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


def _reads_the_environment(node: ast.AST) -> TypeGuard[ast.Attribute | ast.Subscript | ast.ImportFrom]:
    match node:
        case ast.Attribute(value=owner, attr=name):
            return name in ENVIRONMENT and ast.unparse(owner) in ENVIRONMENT_MODULES
        case ast.Subscript(
            value=ast.Call(args=[owner]) | ast.Attribute(attr="__dict__", value=owner) as namespace,
            slice=ast.Constant(value=str(name)),
        ) if not isinstance(namespace, ast.Call) or ast.unparse(namespace.func) in NAMESPACES:
            return name in ENVIRONMENT and ast.unparse(owner) in ENVIRONMENT_MODULES
        case ast.ImportFrom(module=str(module), names=names):
            return module in ENVIRONMENT_MODULES and bool(ENVIRONMENT & {each.name for each in names})
        case _:
            return False


def environment_outside_settings(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if _reads_the_environment(node):
            yield Finding(path, node.lineno, ENVIRONMENT_READ)


def _imported(node: ast.Import | ast.ImportFrom) -> list[str]:
    return [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]


def pinned_examples(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and any(keyword.arg == "max_examples" for keyword in node.keywords):
            yield Finding(path, node.lineno, PINNED_EXAMPLES)


def waits_dropped(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        match node:
            case ast.Expr(value=ast.Call(func=ast.Attribute(attr="wait" | "join"))):
                yield Finding(path, node.lineno, WAIT_DROPPED)
            case _:
                pass


def clocks(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom) and CLOCKS & {n.split(".")[0] for n in _imported(node)}:
            yield Finding(path, node.lineno, CLOCK_IN_A_TEST)


type Tuples = Mapping[str, list[ast.expr]]
type TuplesOf = Callable[[str], Tuples]


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


def module_tuples(tree: ast.Module) -> Tuples:
    return MappingProxyType(_named_tuples(canonical(tree)))


@cache
def first_party_tuples(module: str) -> Tuples:
    path = Path(FIRST_PARTY, *module.split(".")).with_suffix(".py")
    return module_tuples(ast.parse(path.read_text(encoding="utf-8"))) if path.is_file() else MappingProxyType({})


def _caught(node: ast.expr, named: Tuples, tuples_of: TuplesOf, seen: frozenset[str]) -> Iterator[str]:
    if isinstance(node, ast.Tuple):
        for each in node.elts:
            yield from _caught(each, named, tuples_of, seen)
        return
    dotted = ast.unparse(node)
    module, _, name = dotted.rpartition(".")
    held = tuples_of(module) if module else named
    if name not in held:
        yield name
        return
    for each in () if dotted in seen else held[name]:
        yield from _caught(each, held, tuples_of, seen | {dotted})


def _raises_again(handler: ast.ExceptHandler) -> bool:
    return any(isinstance(node, ast.Raise) and node.exc is None for node in handler.body)


def _caught_by(handler: ast.ExceptHandler, named: Tuples, tuples_of: TuplesOf) -> frozenset[str]:
    if handler.type is None:
        return frozenset({"BaseException"})
    return frozenset(_caught(handler.type, named, tuples_of, frozenset()))


def recursion_caught(path: str, tree: ast.Module, tuples_of: TuplesOf = first_party_tuples) -> Iterator[Finding]:
    named = _named_tuples(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try | ast.TryStar):
            continue
        let_through = False
        for handler in node.handlers:
            caught, again = _caught_by(handler, named, tuples_of), _raises_again(handler)
            if caught & RECURSION_AND_BASES and not (let_through or again):
                yield Finding(path, handler.lineno, RECURSION_CAUGHT)
            let_through = let_through or ("RecursionError" in caught and again)


def _pattern(call: ast.Call) -> ast.expr | None:
    given = [*call.args[:1], *(keyword.value for keyword in call.keywords if keyword.arg == "pattern")]
    return given[0] if given else None


def unicode_digits(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) in PATTERN_CALLS:
            match _pattern(node):
                case ast.Constant(value=str(pattern)) if DIGIT_CLASS.search(pattern):
                    yield Finding(path, node.lineno, UNICODE_DIGITS)
                case _:
                    pass


def _flagged(text: str) -> bool:
    return (bool(FLAG.search(text)) and not INSTALLER.search(text)) or bool(COMMAND.search(text))


def flags_outside_interfaces(path: str, tree: ast.Module) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _flagged(node.value):
            yield Finding(path, node.lineno, FLAG_OUTSIDE)


def idioms(path: str, source: str) -> tuple[Finding, ...]:
    tree = canonical(ast.parse(source))
    in_tests, in_sift = path.startswith("tests/"), path.startswith("src/sift/")
    outside_interfaces = in_sift and not path.startswith(INTERFACES)
    return (
        *invisible_characters(path, source),
        *(() if in_tests else banned_methods(path, tree)),
        *(() if in_tests else unicode_digits(path, tree)),
        *(() if in_tests else recursion_caught(path, tree)),
        *(io_in_the_core(path, tree) if path.startswith(CORE) else ()),
        *(exits_outside_interfaces(path, tree) if outside_interfaces else ()),
        *(flags_outside_interfaces(path, tree) if outside_interfaces else ()),
        *(clock_read(path, tree) if outside_interfaces or in_tests else ()),
        *(environment_outside_settings(path, tree) if in_sift and path != SETTINGS else ()),
        *(patching(path, tree) if in_tests else ()),
        *(sift_patched(path, tree) if in_tests else ()),
        *(clocks(path, tree) if in_tests else ()),
        *(pinned_examples(path, tree) if in_tests and path not in EXAMPLE_COUNTS else ()),
        *(waits_dropped(path, tree) if in_tests and path != WAITS else ()),
        *storage_idioms(path, tree),
    )


def main(_argv: list[str]) -> int:
    return report(found(idioms, PYTHON_ROOTS), REMEDY)
