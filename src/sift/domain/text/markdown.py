import re
from collections.abc import Iterable, Iterator

LINE = re.compile(r"[^\n]*\n|[^\n]+")
HEADING = re.compile(r" {0,3}#{1,6}[ \t]")
FENCE = re.compile(r"((?:[ \t]*>)*[ \t]*)(`{3,}|~{3,})")
LIST_INDENT = 4
LINE_BREAKS = re.compile(r"[\r\n]+")


def _beyond(behind: str, line: str) -> bool:
    if not line.strip():
        return False
    if ">" in behind:
        return not line.lstrip(" \t").startswith(">")
    indent = len(behind.expandtabs(LIST_INDENT))
    lead = line[: len(line) - len(line.lstrip(" \t"))]
    return indent >= LIST_INDENT and len(lead.expandtabs(LIST_INDENT)) < indent


def _fence(line: str, open_mark: str | None) -> re.Match[str] | None:
    found = FENCE.match(line)
    if found and not open_mark and found.group(2)[0] == "`" and "`" in line[found.end() :]:
        return None
    return found


def _closes(found: re.Match[str], line: str, open_mark: str) -> bool:
    mark = found.group(2)
    return mark[0] == open_mark[0] and len(mark) >= len(open_mark) and not line[found.end() :].strip()


def outside_code(lines: Iterable[str]) -> Iterator[bool]:
    open_mark: str | None = None
    behind = ""
    for line in lines:
        if open_mark and _beyond(behind, line):
            open_mark = None
        found = _fence(line, open_mark) if "```" in line or "~~~" in line else None
        if found is None:
            yield not open_mark
            continue
        if not open_mark:
            open_mark, behind = found.group(2), found.group(1)
        elif _closes(found, line, open_mark):
            open_mark = None
        yield False


def prose(text: str, first_line: int = 1) -> Iterator[tuple[int, str]]:
    lines = text.split("\n")
    return (
        (number, line)
        for number, (line, outside) in enumerate(zip(lines, outside_code(lines), strict=True), first_line)
        if outside
    )


def _heading_text(line: str) -> str:
    title = line.strip().lstrip("#").strip()
    bare = title.rstrip("#")
    if bare != title and (not bare or bare[-1] in " \t"):
        return bare.strip()
    return title


def heading(text: str) -> str | None:
    return next(
        (title for _, line in prose(text) if HEADING.match(line) and (title := _heading_text(line))),
        None,
    )


def one_line(title: str) -> str:
    return " ".join(part.strip(" \t") for part in LINE_BREAKS.split(title.strip()))
