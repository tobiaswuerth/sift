from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from sift.api.errors import ModelUnavailableError, Problem
from sift.api.events import Reporter
from sift.api.fields import FieldValue
from sift.api.ids import owner
from sift.api.notices import Notice
from sift.api.search import MOST_HITS, SearchHybrid, SearchResult
from sift.app import hits, searching
from sift.app.hits import Retrieved, read
from sift.app.keyword_candidates import Wanted, found
from sift.app.loading import prepared
from sift.app.searching import WATCHING, Open, Searching, Wants, Watching, despite_losses
from sift.app.vector_candidates import QUERY_REMEDY, RESCORING, Rescoring, alike_or_by_keywords, near, query_vector
from sift.domain.arrays import floats, ints
from sift.domain.codec.codes import Floats
from sift.domain.query.expressions import NOTHING, Expression
from sift.domain.query.grammar import MOST_WORDS, any_phrase
from sift.domain.rank.best import Best
from sift.domain.rank.hybrid import blend, named, passage, words
from sift.inference.models import Models
from sift.retrieval.keyword.search import TUNING, Expired

EMPTY_REMEDY = "ask a question, e.g. how dark matter halos form"
WORDS_REMEDY = "ask in fewer words to search by every one of them"
WIDER = 2


@dataclass(frozen=True, slots=True)
class Gathered:
    candidates: list[Retrieved]
    cosines: dict[int, float]
    full: bool
    cut: bool


@dataclass(frozen=True, slots=True)
class QuestionWords:
    expression: Expression
    words: int


def _any_word(text: str) -> QuestionWords:
    found_words = list(dict.fromkeys(words(text)))
    kept = found_words[:MOST_WORDS]
    return QuestionWords(any_phrase(kept) if kept else NOTHING, len(found_words))


def _cut_words(keywords: QuestionWords) -> list[Notice]:
    if keywords.words <= MOST_WORDS:
        return []
    said = f"the question holds {keywords.words:,} words: its keywords are the first {MOST_WORDS:,}"
    return [Notice(Problem.TOO_MANY_WORDS, said, WORDS_REMEDY)]


@dataclass(frozen=True, slots=True)
class HybridAsked:
    request: SearchHybrid
    models: Models
    rescoring: Rescoring
    expired: Expired
    keywords: QuestionWords


@dataclass(frozen=True, slots=True)
class CandidateChunks:
    chunks: list[int]
    full: bool
    cut: bool


def _candidates(stores: Open, asked: HybridAsked, query: Floats | None, wanted: int) -> CandidateChunks:
    by_document = not asked.request.all_passages
    keyword = found(stores, Wanted(wanted, by_document, TUNING, asked.expired), asked.keywords.expression)
    semantic = near(stores, Best(wanted, by_document), query, asked.rescoring) if query is not None else ()
    chunks = list(dict.fromkeys([each.chunk for each in keyword.found] + [each.chunk for each in semantic]))
    return CandidateChunks(chunks, len(keyword.found) >= wanted or len(semantic) >= wanted, keyword.cut)


def _cosines(one: Searching, chunks: Sequence[int], query: Floats, asked: Rescoring) -> dict[int, float]:
    ids, rows = asked.read(one.transaction, np.array(chunks, np.int64))
    scores = rows.astype(np.float32) @ query if len(ids) else np.zeros(0, np.float32)
    return dict(zip(ints(ids), floats(scores), strict=True))


def _embedded(one: Searching) -> bool:
    return one.vectors is not None and bool(len(one.vectors.coded.ids))


def _gathered(stores: Open, query: Floats | None, asked: HybridAsked, wanted: int) -> Gathered:
    found_now = _candidates(stores, asked, query, wanted)
    candidates: list[Retrieved] = []
    cosines: dict[int, float] = {}
    for one in list(stores.searching):
        mine = [chunk for chunk in found_now.chunks if owner(chunk) == one.collection.id]
        if not mine:
            continue
        candidates += stores.guarded(one, partial(read, one, mine, dict.fromkeys(mine, 0.0), whole=True))
        if query is not None and _embedded(one):
            cosines |= stores.guarded(one, partial(_cosines, one, mine, query, asked.rescoring))
    return Gathered(candidates, cosines, found_now.full, found_now.cut)


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
        raise ModelUnavailableError("the reranker gave no score for some passages", QUERY_REMEDY)
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
    held = floats(scores)
    ranked = [replace(candidates[at], hit=replace(candidates[at].hit, score=held[at])) for at in order]
    return ranked if request.all_passages else hits.one_each(_first_of_each(ranked))


def _ranked(stores: Open, asked: HybridAsked, query: Floats | None) -> tuple[list[Retrieved], bool]:
    request = asked.request
    wanted = max(request.candidates, request.wanted + 1)
    while True:
        gathered = despite_losses(partial(_gathered, stores, query, asked, wanted))
        if not gathered.candidates:
            return [], gathered.cut
        scores = _scored(asked.models, request, gathered.candidates, gathered.cosines)
        ranked = _ordered(request, gathered.candidates, scores)
        if len(ranked) > request.wanted or not gathered.full or gathered.cut or wanted > MOST_HITS:
            return ranked, gathered.cut
        wanted = min(wanted * WIDER, MOST_HITS + 1)


def _answered(stores: Open, asked: HybridAsked, report: Reporter) -> hits.Answered:
    request, models = asked.request, asked.models
    if not stores.searching:
        return hits.Answered(())
    embedded = any(map(_embedded, stores.searching))
    prepared(report, (models.prepare, models.prepare_ranker) if embedded else (models.prepare_ranker,))
    query = query_vector(models, request.query) if embedded else None
    if query is not None:
        alike_or_by_keywords(stores, models.name, query)
    ranked, cut = _ranked(stores, asked, query)
    plain = [replace(each, hit=replace(each.hit, text=None)) for each in ranked] if request.compact else ranked
    return hits.shown_of(plain, request, cut=cut)


def _checked(request: SearchHybrid) -> None:
    searching.checked(request)
    searching.checked_query(request.query, EMPTY_REMEDY)


def query(
    home: Path,
    request: SearchHybrid,
    models: Models,
    asked: Rescoring = RESCORING,
    watching: Watching = WATCHING,
) -> SearchResult:
    _checked(request)
    expired = searching.expired_after(request.budget, watching.clock)
    narrowing = searching.narrowed(home, request)
    with ExitStack() as cleanup:
        stores = searching.opened(
            cleanup,
            narrowing,
            Wants(keywords=True, model=models.name, codes=watching.codes, decoded=watching.decoded),
            expired,
        )
        keywords = _any_word(request.query)
        stores.left_out.extend(_cut_words(keywords))
        answered = _answered(stores, HybridAsked(request, models, asked, expired, keywords), watching.report)
        return hits.result(stores, answered, "reranker")
