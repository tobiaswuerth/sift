import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

PYTHON_ROOTS = ("src", "tests", "tools", "noxfile.py")
ROOT = Path(__file__).resolve().parents[1]

type Check = Callable[[str, str], Iterable[Finding]]


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    problem: str


def tracked(roots: Iterable[str], top: Path = ROOT) -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--", *roots],
        cwd=top,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    return sorted(name for name in listed.stdout.split("\0") if name)


def files(roots: Iterable[str], suffixes: tuple[str, ...], top: Path = ROOT) -> list[str]:
    asked = list(roots)
    names = [name for name in tracked(asked, top) if Path(name).suffix in suffixes]
    if not names:
        raise FileNotFoundError(f"git tracks no {' or '.join(suffixes)} file under {', '.join(asked)} in {top}")
    return names


def read(name: str, top: Path = ROOT) -> str:
    return (top / name).read_text(encoding="utf-8")


def sources(roots: Iterable[str]) -> dict[str, str]:
    return {name: read(name) for name in files(roots, (".py",))}


def found(check: Check, roots: Iterable[str], suffixes: tuple[str, ...] = (".py",)) -> tuple[Finding, ...]:
    return tuple(finding for name in files(roots, suffixes) for finding in check(name, read(name)))


def report(findings: Iterable[Finding], remedy: str) -> int:
    shown = [f"{finding.path}:{finding.line}: {finding.problem}" for finding in findings]
    for line in shown:
        print(line)
    if shown:
        print(f"\n{len(shown)} found. {remedy}")
    return 1 if shown else 0
