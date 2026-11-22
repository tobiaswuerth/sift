from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

PYTHON_ROOTS = ("src", "tests", "tools", "noxfile.py")

type Check = Callable[[str, str], Iterable[Finding]]


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    problem: str


def files(roots: Iterable[str], suffixes: tuple[str, ...]) -> Iterator[Path]:
    for root in map(Path, roots):
        if root.is_file():
            yield root
        else:
            yield from sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes)


def found(check: Check, roots: Iterable[str], suffixes: tuple[str, ...] = (".py",)) -> tuple[Finding, ...]:
    return tuple(
        finding
        for path in files(roots, suffixes)
        for finding in check(path.as_posix(), path.read_text(encoding="utf-8"))
    )


def report(findings: Iterable[Finding], remedy: str) -> int:
    shown = [f"{finding.path}:{finding.line}: {finding.problem}" for finding in findings]
    for line in shown:
        print(line)
    if shown:
        print(f"\n{len(shown)} found. {remedy}")
    return 1 if shown else 0
