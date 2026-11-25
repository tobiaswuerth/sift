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
    return collections.status(acting.settings.home, request, audience=acting.audience)


def _remove(acting: Acting, request: RemoveCollection) -> CollectionRemoved:
    return collections.remove(acting.settings.home, request)


def _move(acting: Acting, request: MoveCollections) -> StoresMoved:
    return moving.move(acting.settings.home, request, acting.report)


def _activate(acting: Acting, request: ActivateCollections) -> CollectionsSwitched:
    return collections.activate(acting.settings.home, request)


def _deactivate(acting: Acting, request: DeactivateCollections) -> CollectionsSwitched:
    return collections.deactivate(acting.settings.home, request)


def _get(acting: Acting, request: GetDocuments) -> Documents:
    return documents.documents(acting.settings.home, request, audience=acting.audience)


ADD = Act("add", AddCollection, CollectionAdded | CollectionChanged, _add, writes=True)
REMOVE = Act("rm", RemoveCollection, CollectionRemoved, _remove, writes=True)
MOVE = Act("move", MoveCollections, StoresMoved, _move, writes=True)
ACTIVATE = Act("activate", ActivateCollections, CollectionsSwitched, _activate, writes=True)
DEACTIVATE = Act("deactivate", DeactivateCollections, CollectionsSwitched, _deactivate, writes=True)
STATUS = Act("status", ShowStatus, Status, _status)
GET = Act("get", GetDocuments, Documents, _get)
