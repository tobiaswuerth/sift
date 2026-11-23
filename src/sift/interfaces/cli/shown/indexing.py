from sift.api.embedding import EmbedResult
from sift.api.indexing import IndexResult
from sift.interfaces.cli.shown.outcome import Shown, empty
from sift.interfaces.console.style import Style


def said_as_it_went(result: IndexResult | EmbedResult, platform: str, style: Style) -> Shown:
    return empty(platform, style) if not result.collections else Shown((), ())
