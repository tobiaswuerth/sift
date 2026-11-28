from sift.api.embedding import EmbedResult
from sift.api.indexing import FileIndexed, IndexResult
from sift.interfaces.cli.shown.outcome import Shown, empty
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled

TROUBLED = frozenset({"missing", "unreadable", "left out"})


def _file(each: FileIndexed, style: Style) -> str:
    tone = Tone.WARNING if each.state in TROUBLED else None
    return labelled("file", f"{each.id}: {each.state}" + (f" ({each.note})" if each.note else ""), tone, style)


def said_as_it_went(result: IndexResult | EmbedResult, platform: str, style: Style) -> Shown:
    if not result.collections:
        return empty(platform, style)
    if isinstance(result, EmbedResult):
        return Shown((), ())
    return Shown((), tuple(_file(each, style) for collection in result.collections for each in collection.named))
