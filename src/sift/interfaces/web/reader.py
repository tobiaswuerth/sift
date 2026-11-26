import re
from collections.abc import Sequence
from itertools import accumulate
from urllib.parse import urlencode

from sift.api.documents import Documents, DocumentText
from sift.api.errors import SiftError
from sift.api.links import Links
from sift.interfaces.acts.acting import answered
from sift.interfaces.web.explore import PageReads, refusal
from sift.interfaces.web.html import Html, joined, tag
from sift.interfaces.web.marking import marked, sought
from sift.interfaces.web.view import PAGE, PREFIX, PageView, filtered, href

READER_LINES = 2_000
READER_CHARACTERS = 1 << 20
CUT = "{left:,} more characters in this line: Open the file shows them all"
SHORT = 60
HEADING = re.compile(r"(#{1,6}) ")
SPAN = re.compile(r"([0-9]{1,9})-([0-9]{1,9})")
WEB = ("http://", "https://")


def span_of(lines: str) -> tuple[int, int] | None:
    found = SPAN.fullmatch(lines)
    return (int(found[1]), int(found[2])) if found else None


def _value(view: PageView, key: str, value: object) -> Html:
    shown = ("true" if value else "false") if isinstance(value, bool) else value
    if isinstance(shown, str) and shown.startswith(WEB):
        return tag("a", {"href": shown, "rel": "noopener noreferrer"}, shown)
    if isinstance(shown, str | int | float) and len(str(shown)) <= SHORT:
        return tag("a", {"href": filtered(view, key, shown)}, str(shown))
    return joined(str(shown))


def _fields(view: PageView, document: DocumentText) -> Html:
    rows = list[Html]()
    for key, value in (document.fields or {}).items():
        values = list(value) if isinstance(value, list | tuple) else [value]
        rows.append(
            tag("div", None, tag("dt", None, key), tag("dd", None, [_value(view, key, each) for each in values]))
        )
    return tag("dl", {"class": "fields"}, rows) if rows else Html("")


def _line(number: int, text: str, span: tuple[int, int] | None, view: PageView) -> Html:
    heading = HEADING.match(text)
    kind = f"h{min(len(heading[1]), 3)}" if heading else None
    shown = text[heading.end() :] if heading else text
    hit = span is not None and span[0] <= number <= span[1]
    classes = " ".join(each for each in (kind, "hit" if hit else None) if each) or None
    left = len(shown) - READER_CHARACTERS
    return tag(
        "div",
        {"class": classes, "id": f"l{number}"},
        tag("span", {"class": "n"}, number),
        marked(shown[:READER_CHARACTERS], sought(view.q)),
        tag("span", {"class": "cut"}, CUT.format(left=left)) if left > 0 else "",
    )


def _fitting(lines: Sequence[str]) -> int:
    held = accumulate(len(line) for line in lines)
    return max(sum(1 for each in held if each <= READER_CHARACTERS), min(1, len(lines)))


def _lines(view: PageView, text: str, span: tuple[int, int] | None) -> tuple[Html, int, int, int]:
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    total = len(lines)
    first = view.at or (max(1, span[0] - READER_LINES // 10) if span else 1)
    first = min(first, max(1, total))
    last = first - 1 + _fitting(lines[first - 1 : first - 1 + READER_LINES])
    rows = [_line(number, lines[number - 1], span, view) for number in range(first, last + 1)]
    return tag("div", {"class": "text"}, rows), first, last, total


def _linked(view: PageView, reads: PageReads, document: DocumentText) -> Html:
    try:
        links = answered(reads.read("links", {"id": document.id, "direction": "both", "k": 1}), Links)
    except SiftError:
        return Html("")
    listing = {"q": "", "similar": "", "c": "", "path": "", "f": (), "k": PAGE}
    found = list[Html]()
    if links.into:
        found.append(tag("a", {"href": href(view, to=document.id, by="", **listing)}, "Backlinks"))
    if links.out:
        found.append(tag("a", {"href": href(view, by=document.id, to="", **listing)}, "Links"))
    return joined(found)


def _actions(view: PageView, reads: PageReads, document: DocumentText) -> Html:
    similar = tag("a", {"href": href(view, similar=document.id, q="", k=PAGE)}, "Similar")
    original = tag("a", {"href": f"{PREFIX}/file?{urlencode({'id': document.id})}", "rel": "noopener"}, "Open the file")
    return tag("nav", {"class": "actions"}, similar, _linked(view, reads, document), original)


def _parts(view: PageView, first: int, last: int, total: int) -> Html:
    if first == 1 and last >= total:
        return Html("")
    earlier = tag("a", {"href": href(view, at=max(1, first - READER_LINES))}, "Earlier") if first > 1 else ""
    later = tag("a", {"href": href(view, at=last + 1)}, "Later") if last < total else ""
    return tag("p", {"class": "part"}, f"lines {first:,} to {last:,} of {total:,} ", earlier, " ", later)


def reader(view: PageView, reads: PageReads) -> Html:
    if not view.doc:
        return Html("")
    try:
        documents = answered(reads.read("get", {"documents": [{"id": view.doc}], "fields": ["*"]}), Documents)
    except SiftError as error:
        return tag("article", {"class": "reader"}, refusal(error.message, error.remedy))
    if not documents.found:
        (notice,) = documents.not_read
        return tag("article", {"class": "reader"}, refusal(notice.message, notice.remedy))
    document = documents.found[0]
    text, first, last, total = _lines(view, document.text, span_of(view.lines))
    heading = tag("header", None, tag("h1", None, document.title), tag("p", {"class": "id"}, document.id))
    parts = _parts(view, first, last, total)
    return tag(
        "article",
        {"class": "reader"},
        heading,
        _actions(view, reads, document),
        _fields(view, document),
        parts,
        text,
        parts,
    )
