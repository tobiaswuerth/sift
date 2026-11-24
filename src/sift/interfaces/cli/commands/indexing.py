from types import MappingProxyType

from sift.api.embedding import EmbedCollections, EmbedResult
from sift.api.indexing import IndexCollections, IndexResult, Skipped
from sift.app import embedding, indexing
from sift.app.filing import Readers
from sift.app.scans import Scanning
from sift.interfaces.cli.commands.making import command
from sift.interfaces.cli.commands.models import models_of
from sift.interfaces.cli.context import Context
from sift.interfaces.cli.shown import indexing as shown


def _index(context: Context, request: IndexCollections) -> IndexResult:
    readers = Readers(scanning=Scanning(context.settings.ocr, gpus=context.settings.gpus))
    return indexing.index(context.settings.home, request, context.report, readers=readers)


def _embed(context: Context, request: EmbedCollections) -> EmbedResult:
    return embedding.embed(context.settings.home, request, context.report, models_of(context))


def _some_skipped(result: IndexResult | EmbedResult) -> bool:
    return any(isinstance(each.outcome, Skipped) for each in result.collections)


COMMANDS = MappingProxyType(
    {
        "index": command(IndexCollections, _index, shown.said_as_it_went, _some_skipped),
        "embed": command(EmbedCollections, _embed, shown.said_as_it_went, _some_skipped),
    }
)
