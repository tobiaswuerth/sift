from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from sift.api.collections import NONE_RETITLED, AddCollection, CollectionChanged, Retitled
from sift.api.errors import InvalidRequestError
from sift.api.events import Reporter
from sift.late import late
from sift.storage.catalog import Choices, Collection, set_choices

RETITLING = "sift.app.retitling"
CHANGE_REMEDY = (
    "give what to change alone: where titles come from, whether files linked from outside are read, "
    "or whether agents see it"
)


class Retitling(Protocol):
    def retitle(self, home: Path, one: Collection, wanted: Choices, report: Reporter) -> Retitled: ...


def _otherwise(request: AddCollection, one: Collection) -> bool:
    renamed = request.name is not None and request.name.casefold() != one.name.casefold()
    records = request.id_field is not None or request.title_field is not None or bool(request.text_fields)
    return renamed or request.store is not None or request.nested or records


def changed(home: Path, one: Collection, request: AddCollection, report: Reporter) -> CollectionChanged:
    held = one.choices
    wanted = replace(
        held,
        title_from=request.title_from or held.title_from,
        follow_links=held.follow_links if request.follow_links is None else request.follow_links,
        agents=held.agents if request.agents is None else request.agents,
    )
    if _otherwise(request, one) or wanted == held:
        raise InvalidRequestError(f"{one.folder} is the collection {one.name!r} already", CHANGE_REMEDY)
    if wanted.title_from is held.title_from:
        set_choices(home, one.id, wanted)
        retitled = NONE_RETITLED
    else:
        retitled = late(RETITLING, Retitling).retitle(home, one, wanted, report)
    return CollectionChanged(
        one.name,
        wanted.title_from,
        wanted.follow_links,
        wanted.agents,
        retitled.documents,
        retitled.to_embed,
        index_again=wanted.follow_links != held.follow_links,
    )


if TYPE_CHECKING:
    from sift.app import retitling as _retitling

    _RETITLING: Retitling = _retitling
