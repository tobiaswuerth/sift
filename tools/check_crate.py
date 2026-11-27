import re
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath

from tools.findings import Finding, files, read, report

CRATE = "native/src"
ROOT_MODULE = "lib"
TESTS = "tests"
MOST_LINES = 400
LAYERS = ("numbers", "packed", "lists", "packs", "words", "postings", "segment", "merge", "python")
NAMED = re.compile(r"^[ \t]*(?:pub(?:\([a-z]+\))? )?(?:struct|enum|trait|type) (\w+)", re.MULTILINE)
USED = re.compile(r"\bcrate::(\w+)")
TEXT = re.compile(r'b?"(?:[^"\\]|\\.)*"|b?r(#*)".*?"\1|b?\'(?:[^\'\\]|\\(?:[^u]|u\{[0-9a-fA-F]+\}))\'', re.DOTALL)
COMMENT = re.compile(r"//|/\*")
REMEDY = (
    "Split a module that grew too long by responsibility, give each type of the crate a name of its own, let a "
    f"module use only the layers below it ({' < '.join(LAYERS)}; name a new module's place in LAYERS of "
    "tools/check_crate.py), and say what a comment says with a name, a type or a test."
)


def module_of(path: str) -> str:
    return PurePosixPath(path).relative_to(CRATE).with_suffix("").parts[0]


def _line(source: str, at: int) -> int:
    return source.count("\n", 0, at) + 1


def too_long(path: str, source: str) -> Iterator[Finding]:
    lines = source.count("\n")
    if lines > MOST_LINES:
        yield Finding(path, 1, f"module of {lines} lines, more than {MOST_LINES}")


def renamed(sources: Mapping[str, str]) -> Iterator[Finding]:
    first: dict[str, str] = {}
    for path, source in sources.items():
        for found in NAMED.finditer(source):
            name, line = found.group(1), _line(source, found.start(1))
            if name in first:
                yield Finding(path, line, f"{name} is defined in {first[name]} too")
            else:
                first[name] = f"{path}:{line}"


def climbing(path: str, source: str) -> Iterator[Finding]:
    module = module_of(path)
    if module in (ROOT_MODULE, TESTS):
        return
    if module not in LAYERS:
        yield Finding(path, 1, f"{module} is in no layer of the crate")
        return
    for found in USED.finditer(source):
        used = found.group(1)
        if used not in LAYERS or LAYERS.index(used) > LAYERS.index(module):
            yield Finding(path, _line(source, found.start()), f"{module} uses {used}, a layer above it")


def commented(path: str, source: str) -> Iterator[Finding]:
    code = TEXT.sub(lambda text: "\n" * text.group().count("\n"), source)
    for number, line in enumerate(code.split("\n"), start=1):
        if COMMENT.search(line):
            yield Finding(path, number, "holds a comment")


def crate(sources: Mapping[str, str]) -> list[Finding]:
    each = [
        finding
        for path, source in sources.items()
        for check in (too_long, climbing, commented)
        for finding in check(path, source)
    ]
    return sorted([*each, *renamed(sources)], key=lambda finding: (finding.path, finding.line))


def main(_argv: list[str]) -> int:
    return report(crate({name: read(name) for name in files((CRATE,), (".rs",))}), REMEDY)
