from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial

from sift.api.fields import Fields
from sift.api.ids import ChunkId, owner
from sift.api.notices import Notice
from sift.api.search import Hit, Options, SearchResult
from sift.app.searching import Open, Ranked, ScoredChunk, Searching, despite_losses
from sift.domain.query import fields as shown
from sift.domain.rank.hybrid import standing_out
from sift.domain.rank.works import one_per_work, work_of_fields
from sift.domain.text.kinds import public_id
from sift.storage.documents import fields_by_document, passages
from sift.storage.texts import TextSpan, spans_text

WIDER = 2


@dataclass(frozen=True, slots=True)
class Retrieved:
    chunk: ChunkId
    hit: Hit
    fields: Fields


def read(one: Searching, chunks: Sequence[int], scores: Mapping[int, float], *, whole: bool) -> list[Retrieved]:
    held = passages(one.transaction, [ChunkId(chunk) for chunk in chunks])
    documents = sorted({passage.document for passage in held.values()})
    spans = [TextSpan(passage.document, passage.start, passage.length) for passage in held.values()]
    text: list[str | None] = [*spans_text(one.transaction, spans)] if whole else [None] * len(spans)
    fields = fields_by_document(one.transaction, documents)
    return [
        Retrieved(
            chunk,
            Hit(
                public_id(one.collection.name, passage.path, passage.key),
                one.collection.name,
                passage.path,
                passage.title,
                passage.first_line,
                passage.last_line,
                scores[chunk],
                cut,
            ),
            fields.get(passage.document, {}),
        )
        for (chunk, passage), cut in zip(held.items(), text, strict=True)
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


def presented(ranked: Sequence[Retrieved], request: Options) -> list[tuple[int, Hit]]:
    reads = ranked[: standing_out([each.hit.score for each in ranked])] if request.adaptive else ranked
    every = range(len(reads)) if request.sort is None else shown.order([each.fields for each in reads], request.sort)
    order = every[request.offset :]
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
    wanted = request.wanted
    while True:
        found = rank(wanted)
        held = hits(stores, found.found, request, copy_of)
        if len(held) >= request.wanted or len(found.found) < wanted or found.cut:
            return presented(held[: request.wanted], request), found.cut
        if not more():
            return presented(held, request), True
        wanted *= WIDER


def result(stores: Open, answered: Sequence[tuple[int, Hit]], *, partial: bool) -> SearchResult:
    despite_losses(stores.closed)
    live = {one.collection.id for one in stores.searching}
    found = tuple(hit for chunk, hit in answered if owner(chunk) in live)
    searched, unsearched, left_out = len(stores.searching), tuple(stores.unsearched), tuple(stores.left_out)
    cut = partial or stores.cut
    return SearchResult(found, searched, unsearched, partial=cut, ruled_out=stores.ruled_out, left_out=left_out)


def also_unsearched(found: SearchResult, more: Sequence[Notice]) -> SearchResult:
    said = {each.collection for each in found.unsearched}
    return replace(found, unsearched=(*found.unsearched, *(each for each in more if each.collection not in said)))
