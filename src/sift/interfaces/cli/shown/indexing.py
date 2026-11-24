from sift.api.embedding import EmbedResult
from sift.api.indexing import FileIndexed, IndexResult
from sift.interfaces.cli.shown.outcome import Shown, empty
from sift.interfaces.console.style import Style
from sift.interfaces.console.text import printable


def _file(each: FileIndexed) -> str:
    return printable(f"{each.id}: {each.state}" + (f" ({each.note})" if each.note else ""))


def said_as_it_went(result: IndexResult | EmbedResult, platform: str, style: Style) -> Shown:
    if not result.collections:
        return empty(platform, style)
    if isinstance(result, EmbedResult):
        return Shown((), ())
    return Shown((), tuple(_file(each) for collection in result.collections for each in collection.named))
