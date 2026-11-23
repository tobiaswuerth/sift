import ast
from collections.abc import Iterable, Iterator

from tools.findings import Finding, files, report

ROOT = "src/sift"
REMEDY = "Give each class and type in src/sift a name of its own, so that a name means one thing."


def defined(source: str) -> Iterator[tuple[str, int]]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef):
            yield node.name, node.lineno
        elif isinstance(node, ast.TypeAlias):
            yield node.name.id, node.lineno


def repeated(sources: Iterable[tuple[str, str]]) -> tuple[Finding, ...]:
    first: dict[str, str] = {}
    findings: list[Finding] = []
    for path, source in sources:
        for name, line in defined(source):
            if name in first:
                findings.append(Finding(path, line, f"{name} is defined in {first[name]} too"))
            else:
                first[name] = f"{path}:{line}"
    return tuple(findings)


def main(_argv: list[str]) -> int:
    sources = ((path.as_posix(), path.read_text(encoding="utf-8")) for path in files((ROOT,), (".py",)))
    return report(repeated(sources), REMEDY)
