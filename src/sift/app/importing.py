import hashlib
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from sift.api.collections import AddCollection, Audience, DeactivateCollections, TitleFrom
from sift.api.errors import InvalidRequestError, NotFoundError, Problem, SiftError
from sift.api.events import Advanced, Noted, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.importing import Brought, CollectionImported, HomeImported, ImportHome
from sift.api.indexing import IndexCollections, Skipped
from sift.app.collections import add, chosen, deactivate
from sift.app.filing import READERS, Readers
from sift.app.indexing import index
from sift.domain.arrays import ints
from sift.domain.text.embedding import embedded
from sift.inference.hub import COMMIT
from sift.storage.catalog import Collection, known_collections
from sift.storage.database import Connection, reading, writing
from sift.storage.documents import counts
from sift.storage.former import FormerCollection, FormerStore, former_collections, former_store
from sift.storage.store import connected
from sift.storage.store_files import FORMER_CATALOG
from sift.storage.vector_meta import left
from sift.storage.vectors import Pending, pending, put_vectors, refuse_model
from sift.storage.writer import sole_writer

CARRIED_A_WRITE = 1_000
KEY_BYTES = 8
NEW_STORE = "{stem}-v2.sqlite"
NO_INDEX = "{home} holds no index of sift 1.x: there is no {catalog} in it"
NO_INDEX_REMEDY = "name the folder sift 1.x kept its index in"
TAKEN = "a collection of that name is in this home already, not imported from {home}"
TAKEN_REMEDY = "remove it, or import the other collections by name"
UNPINNED_MODEL = "its vectors are of {model}, which names no whole commit, so they are not carried over"
UNPINNED_REMEDY = "embed the collection again"
UNKNOWN = "the sift 1.x home holds no collection {name}"
UNKNOWN_REMEDY = "name one of its collections, or none to import every one"


def _key(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=KEY_BYTES).digest(), "little")


@dataclass(frozen=True, slots=True, eq=False)
class Keyed:
    keys: NDArray[np.uint64]
    chunks: NDArray[np.int64]

    def found(self, texts: Sequence[str]) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
        asked = np.array([_key(each) for each in texts], np.uint64)
        at = np.minimum(np.searchsorted(self.keys, asked), max(len(self.keys) - 1, 0))
        hit = self.keys[at] == asked if len(self.keys) else np.zeros(len(asked), np.bool_)
        return hit, self.chunks[at[hit]]


def keyed(passages: Iterable[tuple[int, bool, str, str]]) -> Keyed:
    keys, chunks = array("Q"), array("q")
    for chunk, first, title, text in passages:
        keys.append(_key(embedded(title, text, first=first)))
        chunks.append(chunk)
    held_keys, held_chunks = np.frombuffer(keys, np.uint64), np.frombuffer(chunks, np.int64)
    order = np.argsort(held_keys, kind="stable")
    return Keyed(held_keys[order], held_chunks[order])


def _vectors(old: FormerStore, chunks: NDArray[np.int64]) -> NDArray[np.float16]:
    wanted = ints(chunks)
    held = old.vectors(wanted)
    return np.stack([np.frombuffer(held[each], np.float16) for each in wanted])


def _batch(connection: Connection, after: int) -> list[Pending]:
    with reading(connection) as transaction:
        return pending(transaction, after, CARRIED_A_WRITE)


def _carried(connection: Connection, old: FormerStore, model: str, report: Reporter) -> int:
    keys = keyed(old.passages())
    with reading(connection) as transaction:
        refuse_model(transaction, model)
        total = left(transaction)
    report(PhaseStarted(Phase.CARRY, total))
    carried, after, looked = 0, 0, 0
    while waiting := _batch(connection, after):
        hit, chunks = keys.found([embedded(each.title, each.text, first=each.first) for each in waiting])
        if chunks.size:
            ids = np.array([each.chunk for each in waiting], np.int64)[hit]
            with writing(connection) as transaction:
                carried += put_vectors(transaction, ids, _vectors(old, chunks), model).stored
        looked, after = looked + len(waiting), waiting[-1].chunk
        report(Advanced(looked))
    report(PhaseDone(carried))
    return carried


def _target(former: FormerCollection) -> Path:
    return former.store.with_name(NEW_STORE.format(stem=former.store.stem))


def _text(choices: Mapping[str, object], key: str) -> str | None:
    value = choices.get(key)
    return value if isinstance(value, str) else None


def _texts(choices: Mapping[str, object]) -> tuple[str, ...]:
    value = choices.get("text")
    listed = cast("list[object]", value) if isinstance(value, list) else []
    return tuple(each for each in listed if isinstance(each, str))


def _asked(former: FormerCollection) -> AddCollection:
    choices, title_from = former.choices, _text(former.choices, "title_from")
    return AddCollection(
        former.folder,
        former.name,
        str(_target(former)),
        title_from=None if title_from is None else TitleFrom(title_from),
        nested=True,
        follow_links=True if choices.get("links") == "follow" else None,
        id_field=_text(choices, "id"),
        title_field=_text(choices, "title"),
        text_fields=_texts(choices),
    )


def _known(home: Path, name: str) -> Collection | None:
    return next((one for one in known_collections(home) if one.name == name), None)


def _placed(home: Path, former: FormerCollection, source: Path) -> Collection:
    there = _known(home, former.name)
    if there is not None and there.store != _target(former):
        raise InvalidRequestError(TAKEN.format(home=source), TAKEN_REMEDY, code=Problem.TAKEN)
    if there is None:
        add(home, _asked(former))
    (placed,) = chosen(home, (former.name,), Audience.PEOPLE)
    return placed


def _carry_into(one: Collection, former: FormerCollection, report: Reporter) -> tuple[str | None, int]:
    with former_store(former.store) as old:
        model = old.model()
        if model is None:
            return None, 0
        if COMMIT.fullmatch(model.rpartition("@")[2]) is None:
            report(Noted(UNPINNED_MODEL.format(model=model), UNPINNED_REMEDY))
            return model, 0
        with sole_writer(one.store, "an import"), connected(one.store, one.id) as connection:
            return model, _carried(connection, old, model, report)


def _brought(one: Collection, former: FormerCollection, model: str | None) -> Brought:
    with former_store(former.store) as old:
        before = old.counts()
    with connected(one.store, one.id) as connection, reading(connection) as transaction:
        now, waiting = counts(transaction), left(transaction)
    return Brought(
        str(one.store),
        before.documents,
        now.documents,
        before.passages,
        now.chunks,
        before.vectors,
        now.chunks - waiting,
        model,
    )


def _imported(
    home: Path, former: FormerCollection, source: Path, report: Reporter, readers: Readers
) -> Brought | Skipped:
    one = _placed(home, former, source)
    (indexed,) = index(home, IndexCollections((one.name,)), report, readers=readers).collections
    if isinstance(indexed.outcome, Skipped):
        return indexed.outcome
    if not former.active:
        deactivate(home, DeactivateCollections((one.name,)))
    model, _ = _carry_into(one, former, report)
    return _brought(one, former, model)


def _one(home: Path, former: FormerCollection, source: Path, report: Reporter, readers: Readers) -> CollectionImported:
    try:
        return CollectionImported(former.name, _imported(home, former, source, report, readers))
    except SiftError as error:
        report(Noted(error.message, error.remedy, error.code))
        return CollectionImported(former.name, Skipped(error.code, error.message, error.remedy, former.name))


def _chosen(found: tuple[FormerCollection, ...], names: tuple[str, ...]) -> tuple[FormerCollection, ...]:
    held = {one.name: one for one in found}
    unknown = next((name for name in names if name not in held), None)
    if unknown is not None:
        raise NotFoundError(UNKNOWN.format(name=unknown), UNKNOWN_REMEDY, code=Problem.UNKNOWN_COLLECTION)
    return tuple(held[name] for name in names) if names else found


def imported(home: Path, request: ImportHome, report: Reporter, readers: Readers = READERS) -> HomeImported:
    source = home if request.home is None else Path(request.home)
    if not (source / FORMER_CATALOG).is_file():
        raise NotFoundError(NO_INDEX.format(home=source, catalog=FORMER_CATALOG), NO_INDEX_REMEDY, "home")
    chosen = _chosen(former_collections(source), request.collections)
    return HomeImported(str(source), tuple(_one(home, each, source, report, readers) for each in chosen))
