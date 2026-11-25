import re

from sift.domain.rank.hybrid import words
from sift.interfaces.web.html import Html, joined, tag

SNIPPET = 260
OPERATORS = frozenset({"and", "or", "not"})
HEADING_MARKS = re.compile(r"(?<!\S)#{1,6}\s")
ELLIPSIS = "\N{HORIZONTAL ELLIPSIS}"


def pattern_of(query: str) -> re.Pattern[str] | None:
    terms = sorted(set(words(query)) - OPERATORS, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(map(re.escape, terms)) + r")\w*", re.IGNORECASE) if terms else None


def marked(text: str, pattern: re.Pattern[str] | None) -> Html:
    if pattern is None:
        return joined(text)
    parts: list[Html | str] = []
    last = 0
    for found in pattern.finditer(text):
        parts += [text[last : found.start()], tag("mark", None, found[0])]
        last = found.end()
    return joined([*parts, text[last:]])


def snippet(text: str, pattern: re.Pattern[str] | None, title: str = "") -> Html:
    flat = HEADING_MARKS.sub("", " ".join(text.split()))
    flat = flat[len(title) :].lstrip() if title and flat.startswith(title) else flat
    found = pattern.search(flat) if pattern else None
    start = max(0, found.start() - SNIPPET // 4) if found else 0
    piece = flat[start : start + SNIPPET]
    shown = (ELLIPSIS if start else "") + piece + (ELLIPSIS if start + SNIPPET < len(flat) else "")
    return marked(shown, pattern)
