import re
from collections.abc import Iterator

from sift.domain.rank.hybrid import words
from sift.domain.text.tokens import tokens
from sift.interfaces.web.html import Html, joined, tag

SNIPPET = 260
OPERATORS = frozenset({"and", "or", "not"})
HEADING_MARKS = re.compile(r"(?<!\S)#{1,6}\s")
ELLIPSIS = "\N{HORIZONTAL ELLIPSIS}"
WORD = re.compile(r"\w+")

type Sought = tuple[str, ...]


def sought(query: str) -> Sought:
    return tuple(sorted(set(words(query)) - OPERATORS))


def _found(text: str, terms: Sought) -> Iterator[re.Match[str]]:
    for word in WORD.finditer(text) if terms else ():
        if any(token.startswith(terms) for token in tokens(word[0])):
            yield word


def marked(text: str, terms: Sought) -> Html:
    parts: list[Html | str] = []
    last = 0
    for found in _found(text, terms):
        parts += [text[last : found.start()], tag("mark", None, found[0])]
        last = found.end()
    return joined([*parts, text[last:]])


def snippet(text: str, terms: Sought, title: str = "") -> Html:
    flat = HEADING_MARKS.sub("", " ".join(text.split()))
    flat = flat[len(title) :].lstrip() if title and flat.startswith(title) else flat
    found = next(_found(flat, terms), None)
    start = max(0, found.start() - SNIPPET // 4) if found else 0
    piece = flat[start : start + SNIPPET]
    shown = (ELLIPSIS if start else "") + piece + (ELLIPSIS if start + SNIPPET < len(flat) else "")
    return marked(shown, terms)
