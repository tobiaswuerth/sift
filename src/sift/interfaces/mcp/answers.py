from typing import cast

from sift.api.notices import Notice
from sift.api.search import Hit, SearchResult
from sift.api.wire import Json, encoded
from sift.domain.text.wording import counted
from sift.interfaces.acts.acting import answered

type ToolAnswer = dict[str, Json]

FLAGS = ("more", "partial", "loosened")
MOST_TEXT = 100_000
FULL = f"this answer's text is at its {MOST_TEXT:,} characters"
TEXTS_LEFT_OUT = f"the text of {{hits}} is left out, as {FULL}: get them by id and lines"


def said(notice: Notice) -> str:
    told = f"{notice.message}; fix: {notice.remedy}" if notice.remedy else notice.message
    where = notice.id or notice.collection
    return f"{where}: {told}" if where else told


def _hit(hit: Hit, room: int) -> ToolAnswer:
    shown: ToolAnswer = {
        "id": hit.id,
        "title": hit.title,
        "first_line": hit.first_line,
        "last_line": hit.last_line,
        "score": hit.score,
    }
    if hit.text is not None and len(hit.text) <= room:
        shown["text"] = hit.text
    if hit.also:
        shown["also"] = list[Json](hit.also)
    if hit.fields:
        shown["fields"] = encoded(hit.fields)
    return shown


def with_notes(shown: ToolAnswer, notices: tuple[Notice, ...]) -> ToolAnswer:
    return {**shown, "notes": [said(each) for each in notices]} if notices else shown


def hits_of(answer: object, room: int = MOST_TEXT) -> tuple[ToolAnswer, int]:
    result = answered(answer, SearchResult)
    hits, held, cut = list[Json](), 0, 0
    for hit in result.hits:
        one = _hit(hit, room - held)
        if "text" in one:
            held += len(cast("str", one["text"]))
        elif hit.text is not None:
            cut += 1
        hits.append(one)
    shown: ToolAnswer = {"hits": hits, "scored_by": result.scored_by}
    shown |= {flag: True for flag in FLAGS if getattr(result, flag)}
    notes: list[Json] = [said(each) for each in (*result.unsearched, *result.left_out)]
    if cut:
        notes.append(TEXTS_LEFT_OUT.format(hits=counted(cut, "hit")))
    return ({**shown, "notes": notes} if notes else shown), held


def _empty(value: Json) -> bool:
    return value is None or value is False or (isinstance(value, list | dict) and not value)


def _within(value: Json) -> Json:
    if isinstance(value, list):
        return [_within(each) for each in value]
    if isinstance(value, dict):
        held = cast("dict[str, Json]", value)
        return {key: _within(each) for key, each in held.items() if not _empty(each)}
    return value


def compacted(value: dict[str, Json]) -> ToolAnswer:
    return {key: _within(each) for key, each in value.items() if each is not None and each is not False}
