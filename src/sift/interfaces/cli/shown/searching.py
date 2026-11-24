import textwrap

from sift.api.fields import Fields
from sift.api.search import Hit, SearchResult
from sift.domain.text.wording import counted
from sift.interfaces.cli.shown.outcome import Shown, empty, left_out_lines, unsearched_lines, value_text
from sift.interfaces.console.progress import DOT
from sift.interfaces.console.style import Style
from sift.interfaces.console.text import GAP, printable
from sift.interfaces.console.units import duration

SNIPPET_CHARACTERS = 200
SMALLEST_SHOWN = 0.005


def _passage(text: str) -> str:
    return printable(textwrap.shorten(" ".join(text.split()), SNIPPET_CHARACTERS, placeholder=" ..."))


def _fields(fields: Fields) -> str:
    return printable("  ".join(f"{key}: {value_text(value)}" for key, value in fields.items()))


def _score(score: float) -> str:
    return f"{score:.2f}" if score == 0 or abs(score) >= SMALLEST_SHOWN else f"{score:.1e}"


def _hit(rank: int, hit: Hit, ranks: int, scores: int) -> tuple[str, ...]:
    indent = " " * (ranks + scores + 2 * len(GAP))
    head = f"{rank:>{ranks}}{GAP}{_score(hit.score):>{scores}}{GAP}{printable(hit.title)}"
    text = (f"{indent}{_passage(hit.text)}",) if hit.text else ()
    also = (f"{indent}also {', '.join(printable(each) for each in hit.also)}",) if hit.also else ()
    fields = (f"{indent}{_fields(hit.fields)}",) if hit.fields else ()
    return (head, f"{indent}{printable(hit.id)}:{hit.lines}", *also, *fields, *text)


def _footer(result: SearchResult, seconds: float, style: Style, kind: str) -> str:
    found = counted(len(result.hits), "result") if result.hits else "no matches"
    ruled_out = f", {result.ruled_out} ruled out by the links" if result.ruled_out else ""
    searched = f"{counted(result.searched, 'collection')}{ruled_out}"
    separator = f"{GAP}{DOT[0] if style.unicode else DOT[1]}{GAP}"
    partial = ("partial: better matches may be missing",) if result.partial else ()
    return separator.join((found, kind, *partial, searched, duration(seconds)))


def hits(result: SearchResult, seconds: float, platform: str, style: Style, kind: str = "keyword") -> Shown:
    if not result.searched and not result.ruled_out and not result.unsearched:
        return empty(platform, style)
    ranks = len(str(len(result.hits)))
    scores = max((len(_score(hit.score)) for hit in result.hits), default=0)
    lines = [line for rank, hit in enumerate(result.hits, 1) for line in _hit(rank, hit, ranks, scores)]
    problems = unsearched_lines(result.unsearched, style)
    notes = left_out_lines(result.left_out, style)
    return Shown(tuple(lines), (*problems, *notes, _footer(result, seconds, style, kind)))
