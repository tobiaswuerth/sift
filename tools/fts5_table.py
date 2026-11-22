import contextlib
import sqlite3
import textwrap
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tools.findings import Finding, report

MODULE = Path("src/sift/domain/text/fts5_table.py")
TOKENIZER = "unicode61 remove_diacritics 2"
SURROGATES = range(0xD800, 0xE000)
EVERY_POINT = tuple(point for point in range(1, 0x110000) if point not in SURROGATES)
WIDTH = 108


type Tokenizer = Callable[[Iterable[str]], list[list[str]]]


@dataclass(frozen=True, slots=True)
class Table:
    separators: tuple[int, ...]
    folds: tuple[tuple[int, str], ...]


def fts5_tokens(texts: Iterable[str]) -> list[list[str]]:
    with contextlib.closing(sqlite3.connect(":memory:")) as db:
        db.execute(f"CREATE VIRTUAL TABLE t USING fts5(x, tokenize='{TOKENIZER}')")
        db.execute("CREATE VIRTUAL TABLE v USING fts5vocab(t, instance)")
        written = list(texts)
        db.executemany("INSERT INTO t(rowid, x) VALUES (?, ?)", enumerate(written))
        found: list[list[str]] = [[] for _ in written]
        instances = cast("list[tuple[str, int]]", db.execute("SELECT term, doc FROM v ORDER BY doc, offset").fetchall())
        for term, row in instances:
            found[row].append(term)
        return found


def _fold(point: int, between_letters: list[str]) -> str | None:
    match between_letters:
        case ["q", "q"]:
            return None
        case [token] if len(token) in (2, 3) and token.startswith("q") and token.endswith("q"):
            return token[1:-1]
        case _:
            unsayable = "neither a separator nor one character or none, which the table cannot say"
            raise ValueError(f"U+{point:04X} between two letters tokenizes as {between_letters}: {unsayable}")


def probed(points: Sequence[int], tokenized: Tokenizer = fts5_tokens) -> Table:
    between = tokenized(f"q{chr(point)}q" for point in points)
    folds = [(point, _fold(point, inside)) for point, inside in zip(points, between, strict=True)]
    separators = tuple(point for point, fold in folds if fold is None)
    changed = tuple((point, fold) for point, fold in folds if fold is not None and fold != chr(point))
    return Table(separators, changed)


def spans(points: Iterable[int]) -> list[str]:
    found: list[list[int]] = []
    for point in points:
        if found and found[-1][1] == point - 1:
            found[-1][1] = point
        else:
            found.append([point, point])
    return [f"{low:04x}" if low == high else f"{low:04x}-{high:04x}" for low, high in found]


def lines(name: str, words: Iterable[str]) -> list[str]:
    wrapped = textwrap.wrap(" ".join(words), WIDTH, break_long_words=False, break_on_hyphens=False)
    return [f"{name} = (", *(f'    "{line}",' for line in wrapped), ")"]


def module_source(table: Table) -> str:
    folds = (f"{point:04x}={ord(folded):04x}" if folded else f"{point:04x}=" for point, folded in table.folds)
    return "\n".join([*lines("SEPARATORS", spans(table.separators)), *lines("FOLDS", folds), ""])


def compared(module: Path, source: str, *, write: bool) -> int:
    if write:
        module.write_bytes(source.encode("utf-8"))
        return 0
    differs = module.read_text(encoding="utf-8") != source
    found = [Finding(module.as_posix(), 1, "differs from what this SQLite's FTS5 tokenizes")] if differs else []
    return report(found, "Write it anew: python -m tools fts5 --write")


def main(argv: list[str]) -> int:
    return compared(MODULE, module_source(probed(EVERY_POINT)), write=argv == ["--write"])
