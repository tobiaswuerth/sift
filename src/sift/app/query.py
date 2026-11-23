from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from sift.api.errors import InvalidRequestError
from sift.api.fields import FieldValue
from sift.api.ids import owner
from sift.api.search import MOST_HITS, Hit, SearchHybrid, SearchResult
from sift.app import hits, searching
from sift.app.hits import Retrieved, read
from sift.app.keyword_candidates import Wanted, found
from sift.app.searching import Open, Searching, Wants, despite_losses
from sift.app.vector_candidates import QUERY_REMEDY, RESCORING, Rescoring, alike, near, query_vector
from sift.domain.arrays import floats, ints
from sift.domain.codec.codes import Floats
from sift.domain.query.expressions import NOTHING, Expression
from sift.domain.query.grammar import parsed
from sift.domain.rank.best import Best
from sift.domain.rank.hybrid import blend, named, passage, standing_out, words
from sift.inference.models import Models
from sift.retrieval.keyword.search import TUNING, never

EMPTY_REMEDY = "ask a question, e.g. sift query 'how dark matter halos form'"
CANDIDATES_REMEDY = f"ask for 1 to {MOST_HITS:,} candidates; fewer than -k are raised to -k"
WIDER = 2


@dataclass(frozen=True, slots=True)
class Gathered:
    candidates: list[Retrieved]
    cosines: dict[int, float]
    full: bool


def _any_word(text: str) -> Expression:
    found_words = list(dict.fromkeys(words(text)))
    return parsed(" OR ".join(f'"{word}"' for word in found_words)) if found_words else NOTHING


def _candidates(
    stores: Open, request: SearchHybrid, query: Floats | None, asked: Rescoring, wanted: int
) -> tuple[list[int], bool]:
    by_document = not request.all_passages
    keyword = found(stores, Wanted(wanted, by_document, TUNING, never), _any_word(request.query))
    semantic = near(stores, Best(wanted, by_document), query, asked) if query is not None else ()
    chunks = list(dict.fromkeys([each.chunk for each in keyword.found] + [each.chunk for each in semantic]))
    return chunks, len(keyword.found) >= wanted or len(semantic) >= wanted


def _cosines(one: Searching, chunks: Sequence[int], query: Floats, asked: Rescoring) -> dict[int, float]:
    ids, rows = asked.read(one.transaction, np.array(chunks, np.int64))
    scores = rows.astype(np.float32) @ query if len(ids) else np.zeros(0, np.float32)
    return dict(zip(ints(ids), floats(scores), strict=True))


def _embedded(one: Searching) -> bool:
    return one.vectors is not None and bool(len(one.vectors.coded.ids))


def _gathered(stores: Open, query: Floats | None, asked: Rescoring, request: SearchHybrid, wanted: int) -> Gathered:
    chunks, full = _candidates(stores, request, query, asked, wanted)
    candidates: list[Retrieved] = []
    cosines: dict[int, float] = {}
    for one in list(stores.searching):
        mine = [chunk for chunk in chunks if owner(chunk) == one.collection.id]
        if not mine:
            continue
        candidates += stores.guarded(one, partial(read, one, mine, dict.fromkeys(mine, 0.0), whole=True))
        if query is not None and _embedded(one):
            cosines |= stores.guarded(one, partial(_cosines, one, mine, query, asked))
    return Gathered(candidates, cosines, full)


def _values(value: FieldValue) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [text for each in value for text in _values(each)]
    return [] if value is None or isinstance(value, dict) else [str(value)]


def _scored(models: Models, request: SearchHybrid, candidates: list[Retrieved], cosines: dict[int, float]) -> Floats:
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


def _first_of_each(ranked: list[Retrieved]) -> list[Retrieved]:
    seen: set[str] = set()
    kept: list[Retrieved] = []
    for each in ranked:
        if each.hit.id not in seen:
            seen.add(each.hit.id)
            kept.append(each)
    return kept


def _ordered(request: SearchHybrid, candidates: list[Retrieved], scores: Floats) -> list[Retrieved]:
    order = ints(np.lexsort((np.array([each.chunk for each in candidates], np.int64), -scores)))
    rounded = floats(scores)
    ranked = [replace(candidates[at], hit=replace(candidates[at].hit, score=round(rounded[at], 4))) for at in order]
    return ranked if request.all_passages else hits.one_each(_first_of_each(ranked))


def _ranked(
    stores: Open, request: SearchHybrid, models: Models, query: Floats | None, asked: Rescoring
) -> list[Retrieved]:
    wanted = max(request.candidates, request.k)
    while True:
        gathered = despite_losses(partial(_gathered, stores, query, asked, request, wanted))
        if not gathered.candidates:
            return []
        scores = _scored(models, request, gathered.candidates, gathered.cosines)
        ranked = _ordered(request, gathered.candidates, scores)
        if len(ranked) >= request.k or not gathered.full or wanted >= MOST_HITS:
            return ranked
        wanted = min(wanted * WIDER, MOST_HITS)


def _answered(stores: Open, request: SearchHybrid, models: Models, asked: Rescoring) -> list[tuple[int, Hit]]:
    query = query_vector(models, request.query) if any(map(_embedded, stores.searching)) else None
    if query is not None:
        alike(stores, models.name, query)
    ranked = _ranked(stores, request, models, query, asked)[: request.k]
    shown = ranked[: standing_out([each.hit.score for each in ranked])] if request.adaptive else ranked
    plain = [replace(each, hit=replace(each.hit, text=None)) for each in shown] if request.compact else shown
    return hits.presented(plain, request)


def _checked(request: SearchHybrid) -> None:
    searching.checked(request)
    if not request.query.strip():
        raise InvalidRequestError("an empty query finds nothing", EMPTY_REMEDY)
    if not 1 <= request.candidates <= MOST_HITS:
        said = f"--candidates {request.candidates:,} is outside 1 to {MOST_HITS:,}"
        raise InvalidRequestError(said, CANDIDATES_REMEDY)


def query(home: Path, request: SearchHybrid, models: Models, asked: Rescoring = RESCORING) -> SearchResult:
    _checked(request)
    narrowing = searching.narrowed(home, request)
    with ExitStack() as cleanup:
        stores = searching.opened(cleanup, narrowing, Wants(keywords=True, model=models.name))
        return hits.result(stores, _answered(stores, request, models, asked), partial=False)
