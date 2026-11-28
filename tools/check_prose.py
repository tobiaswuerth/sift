import ast
import io
import tokenize
import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tools.findings import PYTHON_ROOTS, ROOT, Finding, files, read, report

ALLOWED = "tools/prose_allowed.toml"
HELP = "Help"
ANNOTATED = "Annotated"
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


def _said(value: ast.expr) -> bool:
    match value:
        case ast.Constant(value=str() | bytes()) | ast.JoinedStr():
            return True
        case ast.Tuple(elts=[*items]) if items:
            return all(isinstance(item, ast.Constant) and item.value is not Ellipsis for item in items)
        case _:
            return False


def _statement(node: ast.AST) -> ast.expr | None:
    match node:
        case ast.Expr(value=value) | ast.Assign(targets=[ast.Name(id="_")], value=value) if _said(value):
            return value
        case _:
            return None


def strings_as_statements(path: str, source: str) -> Iterator[Prose]:
    for node in ast.walk(ast.parse(source)):
        said = _statement(node)
        if isinstance(said, ast.Constant) and isinstance(said.value, str):
            yield Prose(path, said.lineno, "docstring", said.value.strip().split("\n", 1)[0])
        elif said is not None:
            yield Prose(path, said.lineno, "prose", ast.unparse(said))


def _private(name: str) -> bool:
    return name.startswith("_") and name != "_" and not name.endswith("__")


def unread_texts(path: str, tree: ast.Module) -> Iterator[Prose]:
    read = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    for node in tree.body:
        match node:
            case (
                ast.Assign(targets=[ast.Name(id=name)], value=value)
                | ast.AnnAssign(target=ast.Name(id=name), value=ast.expr() as value)
            ) if _private(name) and name not in read and _said(value):
                said = value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else None
                yield Prose(path, node.lineno, "unread", ast.unparse(value) if said is None else said)
            case _:
                pass


def _texts_outside_help(node: ast.expr) -> Iterator[ast.Constant]:
    match node:
        case ast.Call(func=called) if ast.unparse(called).rpartition(".")[2] == HELP:
            return
        case ast.Constant(value=str()):
            yield node
        case _:
            for child in ast.iter_child_nodes(node):
                value = child.value if isinstance(child, ast.keyword) else child
                if isinstance(value, ast.expr):
                    yield from _texts_outside_help(value)


def annotated_texts(path: str, tree: ast.Module) -> Iterator[Prose]:
    for node in ast.walk(tree):
        match node:
            case ast.Subscript(value=annotated, slice=ast.Tuple(elts=[_, *metadata])) if (
                ast.unparse(annotated).rpartition(".")[2] == ANNOTATED
            ):
                for each in metadata:
                    yield from (
                        Prose(path, text.lineno, "annotation", cast("str", text.value))
                        for text in _texts_outside_help(each)
                    )
            case _:
                pass


def prose(path: str, source: str) -> tuple[Prose, ...]:
    tree = ast.parse(source)
    return (
        *comments(path, source),
        *strings_as_statements(path, source),
        *unread_texts(path, tree),
        *annotated_texts(path, tree),
    )


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


def checked(roots: Iterable[str], allowed: Allowed, top: Path = ROOT) -> int:
    written = (piece for name in files(roots, (".py",), top) for piece in prose(name, read(name, top)))
    return report(findings(written, allowed), REMEDY)


def main(_argv: list[str]) -> int:
    return checked(PYTHON_ROOTS, allowed_entries(read(ALLOWED)))
