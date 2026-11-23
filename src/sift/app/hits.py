from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial

from sift.api.fields import Fields
from sift.api.ids import ChunkId, owner
from sift.api.search import Hit, Options, SearchResult, Unsearched
from sift.app.searching import Open, Ranked, ScoredChunk, Searching, despite_losses
from sift.domain.query import fields as shown
from sift.domain.rank.works import one_per_work, work_of_fields
from sift.domain.text.records import public_id
from sift.storage.documents import fields_by_document, passages, texts

WIDER = 2


@dataclass(frozen=True, slots=True)
class Retrieved:
    chunk: int
    hit: Hit
    fields: Fields


def read(one: Searching, chunks: Sequence[int], scores: Mapping[int, float], *, whole: bool) -> list[Retrieved]:
    held = passages(one.transaction, [ChunkId(chunk) for chunk in chunks])
    documents = sorted({passage.document for passage in held.values()})
    text = texts(one.transaction, documents) if whole else {}
    fields = fields_by_document(one.transaction, documents)
    return [
        Retrieved(
            chunk,
            Hit(
                public_id(one.collection.name, passage.path, passage.key),
                passage.title,
                f"{passage.first_line}-{passage.last_line}",
                scores[chunk],
                text[passage.document][passage.start : passage.start + passage.length] if whole else None,
            ),
            fields.get(passage.document, {}),
        )
        for chunk, passage in held.items()
    ]


def _work(each: Retrieved) -> str | None:
    return work_of_fields(each.fields)


def _copied(best: Retrieved, copy: Retrieved) -> Retrieved:
    return replace(best, hit=replace(best.hit, also=(*best.hit.also, copy.hit.id)))


def one_each(reads: Sequence[Retrieved]) -> list[Retrieved]:
    return one_per_work(reads, _work, _copied)


def hits(stores: Open, found: Sequence[ScoredChunk], request: Options, copy_of: str | None = None) -> list[Retrieved]:
    scores = {each.chunk: each.score for each in found}
    by_chunk: dict[int, Retrieved] = {}
    for one in list(stores.searching):
        mine = [each.chunk for each in found if owner(each.chunk) == one.collection.id]
        reads = stores.guarded(one, partial(read, one, mine, scores, whole=not request.compact)) if mine else []
        by_chunk |= {each.chunk: each for each in reads}
    seen = [by_chunk[each.chunk] for each in found if copy_of is None or _work(by_chunk[each.chunk]) != copy_of]
    return seen if request.all_passages else one_each(seen)


def presented(reads: Sequence[Retrieved], request: Options) -> list[tuple[int, Hit]]:
    order = range(len(reads)) if request.sort is None else shown.order([each.fields for each in reads], request.sort)
    wanted = request.fields
    return [
        (reads[at].chunk, replace(reads[at].hit, fields=shown.picked(reads[at].fields, wanted) if wanted else None))
        for at in order
    ]


def always() -> bool:
    return True


def answered(
    stores: Open,
    rank: Callable[[int], Ranked],
    request: Options,
    copy_of: str | None = None,
    more: Callable[[], bool] = always,
) -> tuple[list[tuple[int, Hit]], bool]:
    wanted = request.k
    while True:
        found = rank(wanted)
        held = hits(stores, found.found, request, copy_of)
        if len(held) >= request.k or len(found.found) < wanted or found.cut:
            return presented(held[: request.k], request), found.cut
        if not more():
            return presented(held, request), True
        wanted *= WIDER


def result(stores: Open, answered: Sequence[tuple[int, Hit]], *, partial: bool) -> SearchResult:
    despite_losses(stores.closed)
    live = {one.collection.id for one in stores.searching}
    found = tuple(hit for chunk, hit in answered if owner(chunk) in live)
    searched, unsearched = len(stores.searching), tuple(stores.unsearched)
    return SearchResult(found, searched, unsearched, partial=partial, ruled_out=stores.ruled_out)


def also_unsearched(found: SearchResult, more: Sequence[Unsearched]) -> SearchResult:
    said = {each.collection for each in found.unsearched}
    return replace(found, unsearched=(*found.unsearched, *(each for each in more if each.collection not in said)))
