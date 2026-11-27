import ast
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from tools.findings import ROOT, Finding, report, tracked

PAGE = ROOT / "docs" / "limits.md"
BOUND = re.compile(r"(MOST|FEWEST|LEAST)_\w+|\w+_(SECONDS|BYTES)")
CONSTANT = re.compile(r"^(?:pub(?:\(\w+\))? )?const (\w+): [^=]+ = (.+);$", re.MULTILINE)
CRATE = "native/src"
BAR, ESCAPED_BAR = "|", r"\|"
HEAD = """\
# Limits

Every bound sift sets in its code: the most it reads, holds or sends, the fewest it needs, and how long it waits.
Generated from the code by `python -m tools limits --write`; a test holds it equal to the code.

| Module | Name | Value |
|---|---|---|
"""


@dataclass(frozen=True, slots=True)
class Limit:
    module: str
    name: str
    value: str


def _pairs(target: ast.expr, value: ast.expr) -> Iterator[tuple[ast.expr, ast.expr]]:
    if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple) and len(target.elts) == len(value.elts):
        for inner, written in zip(target.elts, value.elts, strict=True):
            yield from _pairs(inner, written)
    else:
        yield target, value


def _assigned(node: ast.stmt) -> Iterator[tuple[ast.expr, ast.expr]]:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            yield from _pairs(target, node.value)
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        yield from _pairs(node.target, node.value)


def limits_in(module: str, source: str) -> list[Limit]:
    return [
        Limit(module, target.id, ast.get_source_segment(source, value) or ast.unparse(value))
        for node in ast.parse(source).body
        for target, value in _assigned(node)
        if isinstance(target, ast.Name) and BOUND.fullmatch(target.id)
    ]


def limits_in_crate(path: str, source: str) -> list[Limit]:
    module = f"native::{Path(path).stem}"
    found = ((each.group(1), each.group(2)) for each in CONSTANT.finditer(source))
    return [Limit(module, name, value) for name, value in found if BOUND.fullmatch(name)]


def _module(path: str) -> str:
    return ".".join(Path(path).with_suffix("").parts[1:])


def table(found: Iterable[Limit]) -> str:
    rows = (f"| `{each.module}` | `{each.name}` | `{each.value.replace(BAR, ESCAPED_BAR)}` |" for each in found)
    return HEAD + "".join(f"{row}\n" for row in rows)


def compared(page: Path, written: str, *, write: bool) -> int:
    if write:
        page.write_bytes(written.encode("utf-8"))
        return 0
    differs = page.read_text(encoding="utf-8") != written
    found = [Finding(page.relative_to(page.parents[1]).as_posix(), 1, "lists other bounds than the code sets")]
    return report(found if differs else [], "Write it anew: python -m tools limits --write")


def main(argv: list[str]) -> int:
    paths = [path for path in tracked(["src/sift"]) if path.endswith(".py")]
    found = [each for path in paths for each in limits_in(_module(path), (ROOT / path).read_text(encoding="utf-8"))]
    crate = [path for path in tracked([CRATE]) if path.endswith(".rs")]
    native = [each for path in crate for each in limits_in_crate(path, (ROOT / path).read_text(encoding="utf-8"))]
    return compared(PAGE, table([*found, *native]), write=argv == ["--write"])
