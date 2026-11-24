from types import MappingProxyType

from sift.api.collections import Status, Unavailable
from sift.api.documents import Documents
from sift.interfaces.acts.collections import ACTIVATE, ADD, DEACTIVATE, GET, MOVE, REMOVE, STATUS
from sift.interfaces.cli.commands.making import command, needing_collections
from sift.interfaces.cli.shown import collections as shown
from sift.interfaces.cli.shown.documents import documents as shown_documents


def _some_unavailable(found: Status) -> bool:
    return any(isinstance(each.contents, Unavailable) for each in found.collections)


def _some_unfound(found: Documents) -> bool:
    return bool(found.unfound)


COMMANDS = MappingProxyType(
    {
        "add": command(ADD, shown.added),
        "rm": command(REMOVE, shown.removed),
        "move": needing_collections(command(MOVE, shown.moved)),
        "activate": command(ACTIVATE, shown.switched),
        "deactivate": command(DEACTIVATE, shown.switched),
        "status": command(STATUS, shown.status, _some_unavailable),
        "get": needing_collections(command(GET, shown_documents, _some_unfound)),
    }
)
