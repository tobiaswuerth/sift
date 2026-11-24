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
)
from sift.api.documents import Documents, GetDocuments
from sift.app import collections, documents, moving
from sift.interfaces.acts.acting import Act, Acting


def _add(acting: Acting, request: AddCollection) -> CollectionAdded | CollectionChanged:
    return collections.add(acting.settings.home, request, acting.report)


def _status(acting: Acting, request: ShowStatus) -> Status:
    return collections.status(acting.settings.home, request)


def _remove(acting: Acting, request: RemoveCollection) -> CollectionRemoved:
    return collections.remove(acting.settings.home, request)


def _move(acting: Acting, request: MoveCollections) -> StoresMoved:
    return moving.move(acting.settings.home, request, acting.report)


def _activate(acting: Acting, request: ActivateCollections) -> CollectionsSwitched:
    return collections.activate(acting.settings.home, request)


def _deactivate(acting: Acting, request: DeactivateCollections) -> CollectionsSwitched:
    return collections.deactivate(acting.settings.home, request)


def _get(acting: Acting, request: GetDocuments) -> Documents:
    return documents.documents(acting.settings.home, request)


ADD = Act("add", AddCollection, _add)
REMOVE = Act("rm", RemoveCollection, _remove)
MOVE = Act("move", MoveCollections, _move)
ACTIVATE = Act("activate", ActivateCollections, _activate)
DEACTIVATE = Act("deactivate", DeactivateCollections, _deactivate)
STATUS = Act("status", ShowStatus, _status)
GET = Act("get", GetDocuments, _get)
