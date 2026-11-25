import re
import tomllib
from collections.abc import Iterator

from tools.findings import PYTHON_ROOTS, Finding, files, found, read, report

PUBLISHED_ROOTS = ("docs", ".github", "pyproject.toml", *PYTHON_ROOTS)
PUBLISHED_SUFFIXES = (".md", ".py", ".toml", ".txt", ".yml")
GRAPHICS_AND_PROCESSORS = r"\b(?:R[T]X|G[T]X|GeF[o]rce|Quadr[o]|Ra[d]eon|Ryz[e]n|Threadr[i]pper|X[e]on|EP[Y]C)\b"
INTEL_CORE = r"\bCore i[3579]\b|\bi[3579]-[0-9]{4,5}"
MACHINE = re.compile(f"{GRAPHICS_AND_PROCESSORS}|{INTEL_CORE}")
RUNS_ON = re.compile(r"\||!\[|\[!\[|\[[^\]]*\]: ")
REMEDY = (
    "State hardware by kind (a GPU, 8 CPU threads), not by the model of the machine measured. "
    "Wrap prose as the code is wrapped; only tables, code, images and link addresses run on."
)


def machines(path: str, text: str) -> Iterator[Finding]:
    for number, line in enumerate(text.split("\n"), 1):
        if MACHINE.search(line):
            yield Finding(path, number, "names a machine's model")


def long_prose(path: str, text: str, width: int) -> Iterator[Finding]:
    in_code = False
    for number, line in enumerate(text.split("\n"), 1):
        in_code ^= line.lstrip().startswith("```")
        if len(line) > width and not in_code and not RUNS_ON.match(line):
            yield Finding(path, number, f"prose of {len(line)} characters on a line, more than {width}")


def line_length(pyproject: str) -> int:
    document: dict[str, object] = tomllib.loads(pyproject)
    match document:
        case {"tool": {"ruff": {"line-length": int(width)}}}:
            return width
        case _:
            raise TypeError("tool.ruff.line-length in pyproject.toml is a number")


def main(_argv: list[str]) -> int:
    width = line_length(read("pyproject.toml"))
    pages = [name for name in files((".",), (".md",)) if "/" not in name]

    def text(path: str, content: str) -> Iterator[Finding]:
        yield from machines(path, content)
        if path.endswith(".md"):
            yield from long_prose(path, content, width)

    return report(found(text, (*pages, *PUBLISHED_ROOTS), PUBLISHED_SUFFIXES), REMEDY)
