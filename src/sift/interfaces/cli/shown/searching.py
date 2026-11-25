import textwrap
from collections.abc import Mapping
from types import MappingProxyType

from sift.api.fields import Fields
from sift.api.search import Hit, ScoredBy, SearchResult
from sift.domain.text.wording import counted
from sift.interfaces.cli.shown.outcome import Shown, empty, notice_lines, value_text
from sift.interfaces.console.progress import DOT
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import GAP, labelled, printable
from sift.interfaces.console.units import duration

SNIPPET_CHARACTERS = 200
SMALLEST_SHOWN = 0.005
SCORED: Mapping[ScoredBy, str] = MappingProxyType({"keywords": "keyword", "vectors": "semantic", "reranker": "hybrid"})
LOOSENED = "no passage holds every word of the query: these hold some of them"


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


def _footer(result: SearchResult, seconds: float, style: Style) -> str:
    found = (counted(len(result.hits), "result") if result.hits else "no matches") + (
        ", more follow" if result.more else ""
    )
    ruled_out = f", {result.ruled_out} ruled out by the links" if result.ruled_out else ""
    searched = f"{counted(result.searched, 'collection')}{ruled_out}"
    separator = f"{GAP}{DOT[0] if style.unicode else DOT[1]}{GAP}"
    partial = ("partial: better matches may be missing",) if result.partial else ()
    return separator.join((found, SCORED[result.scored_by], *partial, searched, duration(seconds)))


def hits(result: SearchResult, seconds: float, platform: str, style: Style) -> Shown:
    if not result.searched and not result.ruled_out and not result.unsearched:
        return empty(platform, style)
    ranks = len(str(len(result.hits)))
    scores = max((len(_score(hit.score)) for hit in result.hits), default=0)
    lines = [line for rank, hit in enumerate(result.hits, 1) for line in _hit(rank, hit, ranks, scores)]
    problems = notice_lines(result.unsearched, style)
    notes = notice_lines(result.left_out, style, "note")
    loosened = (labelled("note", LOOSENED, Tone.WARNING, style),) if result.loosened else ()
    return Shown(tuple(lines), (*problems, *notes, *loosened, _footer(result, seconds, style)))
