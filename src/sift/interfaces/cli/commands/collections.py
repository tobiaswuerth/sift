from types import MappingProxyType

from sift.api.collections import (
    ActivateCollections,
    AddCollection,
    CollectionAdded,
    CollectionChanged,
    CollectionRemoved,
    CollectionsSwitched,
    DeactivateCollections,
    MoveCollections,
    RemoveCollection,
    ShowStatus,
    Status,
    StoresMoved,
    Unavailable,
)
from sift.api.documents import Documents, GetDocuments
from sift.app import collections, documents, moving
from sift.interfaces.cli.commands.making import command, needing_collections
from sift.interfaces.cli.context import Context
from sift.interfaces.cli.shown import collections as shown
from sift.interfaces.cli.shown.documents import documents as shown_documents


def _add(context: Context, request: AddCollection) -> CollectionAdded | CollectionChanged:
    return collections.add(context.settings.home, request, context.report)


def _status(context: Context, request: ShowStatus) -> Status:
    return collections.status(context.settings.home, request)


def _remove(context: Context, request: RemoveCollection) -> CollectionRemoved:
    return collections.remove(context.settings.home, request)


def _move(context: Context, request: MoveCollections) -> StoresMoved:
    return moving.move(context.settings.home, request, context.report)


def _activate(context: Context, request: ActivateCollections) -> CollectionsSwitched:
    return collections.activate(context.settings.home, request)


def _deactivate(context: Context, request: DeactivateCollections) -> CollectionsSwitched:
    return collections.deactivate(context.settings.home, request)


def _get(context: Context, request: GetDocuments) -> Documents:
    return documents.documents(context.settings.home, request)


def _some_unavailable(found: Status) -> bool:
    return any(isinstance(each.contents, Unavailable) for each in found.collections)


def _some_unfound(found: Documents) -> bool:
    return bool(found.unfound)


COMMANDS = MappingProxyType(
    {
        "add": command(AddCollection, _add, shown.added),
        "rm": command(RemoveCollection, _remove, shown.removed),
        "move": needing_collections(command(MoveCollections, _move, shown.moved)),
        "activate": command(ActivateCollections, _activate, shown.switched),
        "deactivate": command(DeactivateCollections, _deactivate, shown.switched),
        "status": command(ShowStatus, _status, shown.status, _some_unavailable),
        "get": needing_collections(command(GetDocuments, _get, shown_documents, _some_unfound)),
    }
)
