from sift.api.embedding import EmbedResult
from sift.api.indexing import IndexResult
from sift.interfaces.cli.shown.outcome import Shown, empty
from sift.interfaces.console.style import Style


def indexed(result: IndexResult, platform: str, style: Style) -> Shown:
    return empty(platform, style) if not result.collections else Shown((), ())


def embedded(result: EmbedResult, platform: str, style: Style) -> Shown:
    return empty(platform, style) if not result.collections else Shown((), ())
