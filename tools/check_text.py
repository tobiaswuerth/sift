import re
import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

from tools.findings import PYTHON_ROOTS, Finding, files, found, read, report

PUBLISHED_ROOTS = ("docs", ".github", "pyproject.toml", *PYTHON_ROOTS)
PUBLISHED_SUFFIXES = (".md", ".py", ".toml", ".txt", ".yml")
GRAPHICS_AND_PROCESSORS = r"\b(?:GeF[o]rce|Quadr[o]|Ra[d]eon|Ryz[e]n|Threadr[i]pper|X[e]on|EP[Y]C)\b|\b[RG][T]X"
DATACENTER = r"\b(?:A[1-4]0{1,2}|[HB][12]00|G[H]200|[VP]100|L4[0]S?|M[I][0-9]{3}X?)\b"
APPLE = r"\bM[1-9] (?:Pro|Max|Ultra)\b|\bApple M[1-9]\b"
INTEL_CORE = r"\bCore (?:i[3579]|Ultra [579])\b|\bi[3579]-[0-9]{4,5}"
MACHINE = re.compile(f"{GRAPHICS_AND_PROCESSORS}|{DATACENTER}|{APPLE}|{INTEL_CORE}")
OPENING = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")
CLOSING = re.compile(r" {0,3}(`{3,}|~{3,})[ \t]*")
INDENTED = re.compile(r" {4}|\t")
LIST_ITEM = re.compile(r" {0,3}(?:[-+*]|[0-9]{1,9}[.)])(?: |$)")
RUNS_ON = re.compile(r"\||!\[|\[!\[|\[[^\]]*\]: ")
REMEDY = (
    "State hardware by kind (a GPU, 8 CPU threads), not by the model of the machine measured. "
    "Wrap prose as the code is wrapped; only tables, code, images and link addresses run on."
)


def machines(path: str, text: str) -> Iterator[Finding]:
    for number, line in enumerate(text.split("\n"), 1):
        if MACHINE.search(line):
            yield Finding(path, number, "names a machine's model")


@dataclass(frozen=True, slots=True)
class _Reading:
    fence: str = ""
    listed: bool = False
    after_break: bool = True


def _fenced(reading: _Reading, line: str) -> _Reading:
    closing = CLOSING.fullmatch(line)
    closed = (
        closing is not None and closing.group(1)[0] == reading.fence[0] and len(closing.group(1)) >= len(reading.fence)
    )
    return _Reading(listed=reading.listed) if closed else reading


def _opened(line: str) -> str:
    opening = OPENING.fullmatch(line)
    if opening is None or (opening.group(1)[0] == "`" and "`" in opening.group(2)):
        return ""
    return opening.group(1)


def _read(reading: _Reading, line: str) -> tuple[_Reading, bool]:
    if reading.fence:
        return _fenced(reading, line), True
    if fence := _opened(line):
        return replace(reading, fence=fence), True
    if not line.strip():
        return replace(reading, after_break=True), False
    if INDENTED.match(line):
        code = reading.after_break and not reading.listed
        return replace(reading, after_break=code), code
    listed = bool(LIST_ITEM.match(line)) or (reading.listed and not reading.after_break)
    return _Reading(listed=listed, after_break=False), False


def code_lines(lines: Iterable[str]) -> Iterator[bool]:
    reading = _Reading()
    for line in lines:
        reading, code = _read(reading, line)
        yield code


def long_prose(path: str, text: str, width: int) -> Iterator[Finding]:
    lines = text.split("\n")
    for number, (line, in_code) in enumerate(zip(lines, code_lines(lines), strict=True), 1):
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
