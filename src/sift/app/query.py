from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from sift.api.errors import InvalidRequestError
from sift.api.fields import Fields, FieldValue
from sift.api.ids import ChunkId, owner
from sift.api.search import MOST_HITS, Hit, SearchHybrid, SearchResult
from sift.app import searching
from sift.app.search import Keywords, Wanted, found
from sift.app.vsearch import (
    ASKED,
    QUERY_REMEDY,
    Asked,
    Held,
    alike,
    embedded_by,
    near,
    no_vectors,
    query_vector,
)
from sift.domain.arrays import floats, ints
from sift.domain.codec.codes import Floats
from sift.domain.query.expressions import NOTHING, Expression
from sift.domain.query.grammar import parsed
from sift.domain.rank.best import Best
from sift.domain.rank.hybrid import blend, named, passage, standing_out, words, work_of
from sift.domain.rank.works import one_per_work
from sift.domain.text.records import record_path
from sift.inference.models import Models
from sift.retrieval.keyword.lists import opened
from sift.retrieval.keyword.search import TUNING, never
from sift.storage.database import Reading
from sift.storage.documents import fields_by_document, passages, texts
from sift.storage.keywords import refuse_stale
from sift.storage.passing import Passing

EMPTY_REMEDY = "ask a question, e.g. sift query 'how dark matter halos form'"
CANDIDATES_REMEDY = f"ask for 1 to {MOST_HITS:,} candidates; fewer than -k are raised to -k"


@dataclass(frozen=True, slots=True)
class Both:
    keywords: Keywords
    vectors: Held


@dataclass(frozen=True, slots=True)
class Candidate:
    chunk: int
    hit: Hit
    fields: Fields


type Kept = searching.Kept[Both]


def _both(model: str, transaction: Reading, allowed: Passing | None) -> Both:
    refuse_stale(transaction)
    vectors = embedded_by(model, transaction)
    held = no_vectors() if vectors is None else vectors
    return Both(Keywords(opened(transaction), allowed), replace(held, allowed=allowed))


def _keywords(both: Both) -> Keywords:
    return both.keywords


def _vectors(both: Both) -> Held:
    return both.vectors


def _any_word(text: str) -> Expression:
    found_words = list(dict.fromkeys(words(text)))
    return parsed(" OR ".join(f'"{word}"' for word in found_words)) if found_words else NOTHING


def _candidates(kept: Kept, request: SearchHybrid, query: Floats | None, asked: Asked) -> list[int] | None:
    wanted = max(request.candidates, request.k)
    by_document = not request.all_passages
    keyword = found(kept, _keywords, Wanted(wanted, by_document, TUNING, never), _any_word(request.query))
    if keyword is None:
        return None
    semantic = near(kept, _vectors, Best(wanted, by_document), query, asked) if query is not None else ()
    if semantic is None:
        return None
    return list(dict.fromkeys([each.chunk for each in keyword.found] + [each.chunk for each in semantic]))


def _read(one: searching.Searching[Both], chunks: Sequence[int], request: SearchHybrid) -> list[Candidate]:
    held_passages = passages(one.transaction, [ChunkId(chunk) for chunk in chunks])
    documents = sorted({each.document for each in held_passages.values()})
    whole, fields = texts(one.transaction, documents), fields_by_document(one.transaction, documents)
    return [
        Candidate(
            chunk,
            Hit(
                f"{one.collection.name}/{record_path(each.path, each.key)}",
                each.title,
                f"{each.first_line}-{each.last_line}",
                0.0,
                whole[each.document][each.start : each.start + each.length],
                fields=fields[each.document] if searching.fielded(request) else None,
            ),
            fields[each.document],
        )
        for chunk, each in held_passages.items()
    ]


def _cosines(one: searching.Searching[Both], chunks: Sequence[int], query: Floats, asked: Asked) -> dict[int, float]:
    ids, rows = asked.read(one.transaction, np.array(chunks, np.int64))
    scores = rows.astype(np.float32) @ query if len(ids) else np.zeros(0, np.float32)
    return dict(zip(ints(ids), floats(scores), strict=True))


def _gathered(
    kept: Kept, chunks: list[int], query: Floats | None, asked: Asked, request: SearchHybrid
) -> tuple[list[Candidate], dict[int, float]] | None:
    candidates: list[Candidate] = []
    cosines: dict[int, float] = {}
    for one in list(kept.searching):
        mine = [chunk for chunk in chunks if owner(chunk) == one.collection.id]
        if not mine:
            continue
        read = kept.guarded(one, partial(_read, one, mine, request))
        if read is None:
            return None
        near_by = (
            {}
            if query is None or not len(one.held.vectors.coded.ids)
            else kept.guarded(one, partial(_cosines, one, mine, query, asked))
        )
        if near_by is None:
            return None
        candidates += read
        cosines |= near_by
    return candidates, cosines


def _values(value: FieldValue) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [text for each in value for text in _values(each)]
    return [] if value is None or isinstance(value, dict) else [str(value)]


def _scored(models: Models, request: SearchHybrid, candidates: list[Candidate], cosines: dict[int, float]) -> Floats:
    asked = frozenset(words(request.query))
    names = [
        named([text for key, value in each.fields.items() if key != "title" for text in _values(value)], asked)
        for each in candidates
    ]
    shown = [
        passage(each.hit.title, found_names, each.hit.text or "")
        for each, found_names in zip(candidates, names, strict=True)
    ]
    rerank = models.rerank(request.query, shown).astype(np.float64)
    if not bool(np.isfinite(rerank).all()):
        raise InvalidRequestError("the reranker gave no score for some passages", QUERY_REMEDY)
    cosine = np.array([cosines.get(each.chunk, np.nan) for each in candidates], np.float64)
    return blend(rerank, cosine, np.array([bool(found_names) for found_names in names], np.bool_)).astype(np.float32)


def _first_of_each(ranked: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    kept: list[Candidate] = []
    for each in ranked:
        if each.hit.id not in seen:
            seen.add(each.hit.id)
            kept.append(each)
    return kept


def _work(each: Candidate) -> str | None:
    return work_of(each.fields.get("doi"))


def _copied(best: Candidate, copy: Candidate) -> Candidate:
    return replace(best, hit=replace(best.hit, also=(*best.hit.also, copy.hit.id)))


def _ordered(request: SearchHybrid, candidates: list[Candidate], scores: Floats) -> list[Candidate]:
    order = ints(np.lexsort((np.array([each.chunk for each in candidates], np.int64), -scores)))
    rounded = floats(scores)
    ranked = [replace(candidates[at], hit=replace(candidates[at].hit, score=round(rounded[at], 4))) for at in order]
    if not request.all_passages:
        ranked = _first_of_each(ranked)
        ranked = one_per_work(ranked, _work, _copied)
    ranked = ranked[: request.k]
    return ranked[: standing_out([each.hit.score for each in ranked])] if request.adaptive else ranked


def _answered(kept: Kept, request: SearchHybrid, models: Models, asked: Asked) -> list[tuple[int, Hit]]:
    query = (
        query_vector(models, request.query)
        if any(len(each.held.vectors.coded.ids) for each in kept.searching)
        else None
    )
    if query is not None:
        alike(kept, _vectors, models.name, query)
    while True:
        chunks = _candidates(kept, request, query, asked)
        gathered = None if chunks is None else _gathered(kept, chunks, query, asked, request)
        if gathered is None:
            continue
        candidates, cosines = gathered
        if not candidates:
            return []
        ranked = _ordered(request, candidates, _scored(models, request, candidates, cosines))
        answered = [(each.chunk, replace(each.hit, text=None) if request.compact else each.hit) for each in ranked]
        return searching.presented(answered, request)


def _checked(request: SearchHybrid) -> None:
    searching.checked(request)
    if not request.query.strip():
        raise InvalidRequestError("an empty query finds nothing", EMPTY_REMEDY)
    if not 1 <= request.candidates <= MOST_HITS:
        said = f"--candidates {request.candidates:,} is outside 1 to {MOST_HITS:,}"
        raise InvalidRequestError(said, CANDIDATES_REMEDY)


def query(home: Path, request: SearchHybrid, models: Models, asked: Asked = ASKED) -> SearchResult:
    _checked(request)
    narrowing = searching.narrowed(home, request)
    with ExitStack() as cleanup:
        kept = searching.opened(cleanup, narrowing, partial(_both, models.name))
        return searching.result(kept, _answered(kept, request, models, asked), partial=False)
