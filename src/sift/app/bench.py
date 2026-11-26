import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from sift.api.bench import MeasureSearches, Mode, ModeMeasured, SearchesMeasured
from sift.api.documents import GetDocuments, WantedDocument
from sift.api.errors import InvalidRequestError
from sift.api.events import Advanced, Phase, PhaseDone, PhaseStarted, Reporter
from sift.api.search import Hit, SearchHybrid, SearchKeywords, SearchVectors
from sift.app import query, search, vsearch
from sift.app.documents import documents
from sift.app.kept_decoded import KeptDecoded
from sift.app.loading import prepared
from sift.app.resident import ResidentCodes
from sift.app.searching import Watching
from sift.domain.rank.measures import CUT, MOST_GRADE, Grades, found, ndcg, recall, reciprocal_rank
from sift.domain.text.json_values import deeper_than
from sift.inference.models import Models

MOST_FILE_BYTES = 64 << 20
FIELD = "file"
SHAPE = '[{"query": "...", "relevant": ["collection/path", ...]}, ...]'
NO_CASES = f"the file holds no list of cases; write {SHAPE}"
CASE_REMEDY = (
    "give each case a query and the ids of the documents it should find: a list of them, "
    f"or each with a grade above 0, up to {MOST_GRADE}"
)
UNREAD_REMEDY = "give the path of a file this user may read"
TOO_LONG = f"the file is longer than {MOST_FILE_BYTES:,} bytes"
TOO_LONG_REMEDY = "split the cases into several files"
MOST_NESTING = 8
NESTED = "is nested deeper than a list of cases"


@dataclass(frozen=True, slots=True)
class Case:
    query: str
    relevant: Grades


@dataclass(frozen=True, slots=True)
class Searches:
    home: Path
    models: Models
    watching: Watching
    collections: tuple[str, ...]
    report: Reporter


def _read(path: Path) -> object:
    try:
        with path.open("rb") as held:
            raw = held.read(MOST_FILE_BYTES + 1)
    except OSError as error:
        raise InvalidRequestError(f"{path} cannot be read: {error.strerror or error}", UNREAD_REMEDY, FIELD) from None
    if len(raw) > MOST_FILE_BYTES:
        raise InvalidRequestError(f"{path}: {TOO_LONG}", TOO_LONG_REMEDY, FIELD)
    try:
        text = raw.decode(json.detect_encoding(raw))
        if deeper_than(text, MOST_NESTING):
            raise InvalidRequestError(f"{path} {NESTED}", f"write {SHAPE}", FIELD)
        return cast("object", json.loads(text))
    except ValueError:
        raise InvalidRequestError(f"{path} is no JSON", f"write {SHAPE}", FIELD) from None


def _grade(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        grade = float(value)
    except OverflowError:
        return None
    return grade if 0 <= grade <= MOST_GRADE else None


def _grades(relevant: object) -> dict[str, float] | None:
    if isinstance(relevant, list):
        listed = cast("list[object]", relevant)
        ids = [each for each in listed if isinstance(each, str)]
        return dict.fromkeys(ids, 1.0) if ids and len(ids) == len(listed) else None
    if not isinstance(relevant, dict):
        return None
    given = cast("dict[str, object]", relevant)
    graded = {name: _grade(value) for name, value in given.items()}
    kept = {name: grade for name, grade in graded.items() if grade}
    return kept if kept and None not in graded.values() else None


def _refused(at: int, said: str) -> InvalidRequestError:
    return InvalidRequestError(f"case {at:,} {said}", CASE_REMEDY, FIELD)


def _named(home: Path, at: int, grades: Mapping[str, float]) -> Grades:
    asked = tuple(grades)
    read = documents(home, GetDocuments(tuple(WantedDocument(each) for each in asked)))
    if read.not_read:
        raise _refused(at, f"names {read.not_read[0].message}")
    return MappingProxyType({found_.id: grades[each] for each, found_ in zip(asked, read.found, strict=True)})


def _case(home: Path, at: int, written: object) -> Case:
    entry = cast("dict[str, object]", written) if isinstance(written, dict) else {}
    question, grades = entry.get("query"), _grades(entry.get("relevant"))
    if not isinstance(question, str) or not question.strip() or grades is None:
        raise _refused(at, "needs a query and the documents it should find")
    return Case(question, _named(home, at, grades))


def cases_of(home: Path, path: Path) -> tuple[Case, ...]:
    read = _read(path)
    if not isinstance(read, list) or not read:
        raise InvalidRequestError(f"{path}: {NO_CASES}", f"write {SHAPE}", FIELD)
    return tuple(_case(home, at, each) for at, each in enumerate(cast("list[object]", read), 1))


def _hits(mode: Mode, asked: Searches, question: str) -> tuple[Hit, ...]:
    home, collections, watching = asked.home, asked.collections, asked.watching
    if mode is Mode.KEYWORD:
        return search.search(home, SearchKeywords(question, collections=collections, compact=True), watching).hits
    if mode is Mode.SEMANTIC:
        request = SearchVectors(question, collections=collections, compact=True)
        return vsearch.vsearch(home, request, asked.models, watching=watching).hits
    return query.query(
        home, SearchHybrid(question, collections=collections, compact=True), asked.models, watching=watching
    ).hits


def _scored(hits: Sequence[Hit], case: Case) -> tuple[float, float, float]:
    said = found([(hit.id, *hit.also) for hit in hits], case.relevant)
    return (
        ndcg(said.ranked, said.relevant, CUT),
        reciprocal_rank(said.ranked, said.relevant, CUT),
        recall(said.ranked, said.relevant, CUT),
    )


def _mode(mode: Mode, asked: Searches, cases: Sequence[Case], clock: Callable[[], float]) -> ModeMeasured:
    _hits(mode, asked, cases[0].query)
    report = asked.report
    report(PhaseStarted(Phase.MEASURE, len(cases)))
    started, scores = clock(), list[tuple[float, float, float]]()
    for case in cases:
        scores.append(_scored(_hits(mode, asked, case.query), case))
        report(Advanced(len(scores)))
    elapsed = clock() - started
    report(PhaseDone(len(scores)))
    means = [sum(each[at] for each in scores) / len(scores) for at in range(3)]
    return ModeMeasured(mode, *means, ms_per_query=1000 * elapsed / len(cases))


def measured(
    home: Path, request: MeasureSearches, models: Models, report: Reporter, clock: Callable[[], float]
) -> SearchesMeasured:
    cases = cases_of(home, Path(request.file))
    prepared(report, (models.prepare, models.prepare_ranker))
    watching = Watching(codes=ResidentCodes(), decoded=KeptDecoded())
    asked = Searches(home, models, watching, request.collections, report)
    return SearchesMeasured(len(cases), tuple(_mode(each, asked, cases, clock) for each in Mode))
