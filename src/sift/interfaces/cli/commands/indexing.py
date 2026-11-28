from collections.abc import Mapping
from types import MappingProxyType

from sift.api.embedding import EmbedResult
from sift.api.indexing import IndexResult, Skipped
from sift.interfaces.acts.indexing import EMBED, INDEX
from sift.interfaces.cli.commands.making import command
from sift.interfaces.cli.context import Command
from sift.interfaces.cli.shown import indexing as shown


def _some_skipped(result: IndexResult | EmbedResult) -> bool:
    return any(isinstance(each.outcome, Skipped) for each in result.collections)


COMMANDS: Mapping[str, Command] = MappingProxyType(
    {
        "index": command(INDEX, shown.said_as_it_went, _some_skipped),
        "embed": command(EMBED, shown.said_as_it_went, _some_skipped),
    }
)
