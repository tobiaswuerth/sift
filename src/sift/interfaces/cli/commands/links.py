from types import MappingProxyType

from sift.api.links import Links, ListUnresolved, ShowLinks, Unresolved
from sift.app import links
from sift.interfaces.cli.commands.making import command, needing_collections
from sift.interfaces.cli.context import Context
from sift.interfaces.cli.shown import links as shown


def _links(context: Context, request: ShowLinks) -> Links:
    return links.links(context.settings.home, request)


def _unresolved(context: Context, request: ListUnresolved) -> Unresolved:
    return links.unresolved(context.settings.home, request)


def _some_unread(found: Links | Unresolved) -> bool:
    return bool(found.unsearched)


COMMANDS = MappingProxyType(
    {
        "links": needing_collections(command(ShowLinks, _links, shown.links, _some_unread)),
        "unresolved": needing_collections(command(ListUnresolved, _unresolved, shown.unresolved, _some_unread)),
    }
)
