import ast
import io
import tokenize
import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from tools.findings import PYTHON_ROOTS, Finding, files, report

ALLOWED = "tools/prose_allowed.toml"
REMEDY = (
    "Say it with a name, a type, a test or the structure instead. Only what no code can express may stay, "
    f"with the owner's yes, as an [[allowed]] entry in {ALLOWED} naming path, text and reason."
)

type Allowed = frozenset[tuple[str, str]]


@dataclass(frozen=True, slots=True)
class Prose:
    path: str
    line: int
    kind: str
    text: str


def comments(path: str, source: str) -> Iterator[Prose]:
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            yield Prose(path, token.start[0], "comment", token.string)


def strings_as_statements(path: str, source: str) -> Iterator[Prose]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            yield Prose(path, node.lineno, "docstring", node.value.value.strip().split("\n", 1)[0])


def prose(path: str, source: str) -> tuple[Prose, ...]:
    return (*comments(path, source), *strings_as_statements(path, source))


def allowed_entries(text: str) -> Allowed:
    document: dict[str, object] = tomllib.loads(text)
    return frozenset(map(_entry, _entries(document.get("allowed", []))))


def _entries(value: object) -> Iterator[object]:
    if not isinstance(value, list):
        raise TypeError(f"{ALLOWED} holds [[allowed]] tables")
    yield from value


def _entry(entry: object) -> tuple[str, str]:
    match entry:
        case {"path": str(path), "text": str(said), "reason": str()}:
            return path, said
        case _:
            raise ValueError(f"each [[allowed]] entry needs path, text and reason as strings, got {entry!r}")


def findings(written: Iterable[Prose], allowed: Allowed) -> tuple[Finding, ...]:
    kept = tuple(written)
    present = {(piece.path, piece.text) for piece in kept}
    unallowed = (
        Finding(piece.path, piece.line, f"{piece.kind}: {piece.text}")
        for piece in kept
        if (piece.path, piece.text) not in allowed
    )
    stale = (
        Finding(ALLOWED, 1, f"allows what {path} no longer holds: {text}") for path, text in sorted(allowed - present)
    )
    return (*unallowed, *stale)


def checked(roots: Iterable[str], allowed: Allowed) -> int:
    written = (
        piece for path in files(roots, (".py",)) for piece in prose(path.as_posix(), path.read_text(encoding="utf-8"))
    )
    return report(findings(written, allowed), REMEDY)


def main(_argv: list[str]) -> int:
    return checked(PYTHON_ROOTS, allowed_entries(Path(ALLOWED).read_text(encoding="utf-8")))
