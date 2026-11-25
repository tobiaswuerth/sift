from typing import cast

from sift.api.notices import Notice
from sift.api.search import Hit, SearchResult
from sift.api.wire import Json, encoded
from sift.interfaces.acts.acting import answered

type ToolAnswer = dict[str, Json]

FLAGS = ("more", "partial", "loosened")


def said(notice: Notice) -> str:
    told = f"{notice.message}; fix: {notice.remedy}" if notice.remedy else notice.message
    where = notice.id or notice.collection
    return f"{where}: {told}" if where else told


def _hit(hit: Hit) -> ToolAnswer:
    shown: ToolAnswer = {
        "id": hit.id,
        "title": hit.title,
        "first_line": hit.first_line,
        "last_line": hit.last_line,
        "score": hit.score,
    }
    if hit.text is not None:
        shown["text"] = hit.text
    if hit.also:
        shown["also"] = list[Json](hit.also)
    if hit.fields is not None:
        shown["fields"] = encoded(hit.fields)
    return shown


def with_notes(shown: ToolAnswer, notices: tuple[Notice, ...]) -> ToolAnswer:
    return {**shown, "notes": [said(each) for each in notices]} if notices else shown


def hits_of(answer: object) -> ToolAnswer:
    result = answered(answer, SearchResult)
    shown: ToolAnswer = {"hits": [_hit(hit) for hit in result.hits]}
    shown |= {flag: True for flag in FLAGS if getattr(result, flag)}
    return with_notes(shown, (*result.unsearched, *result.left_out))


def compacted(value: Json) -> Json:
    if isinstance(value, list):
        return [compacted(each) for each in value]
    if isinstance(value, dict):
        held = cast("dict[str, Json]", value)
        return {key: compacted(each) for key, each in held.items() if each is not None and each is not False}
    return value
