from types import MappingProxyType

from sift.api.links import Links, Unresolved
from sift.interfaces.acts.links import LINKS, UNRESOLVED
from sift.interfaces.cli.commands.making import command, needing_collections
from sift.interfaces.cli.shown import links as shown


def _some_unread(found: Links | Unresolved) -> bool:
    return bool(found.unsearched)


COMMANDS = MappingProxyType(
    {
        "links": needing_collections(command(LINKS, shown.links, _some_unread)),
        "unresolved": needing_collections(command(UNRESOLVED, shown.unresolved, _some_unread)),
    }
)
