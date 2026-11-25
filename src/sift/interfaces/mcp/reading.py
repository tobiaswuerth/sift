from collections.abc import Mapping
from typing import cast

from sift.api.documents import Documents, DocumentText, GetDocuments
from sift.api.wire import Json, encoded
from sift.interfaces.acts.acting import answered
from sift.interfaces.mcp.answers import ToolAnswer, said
from sift.interfaces.mcp.schemas import schema_of
from sift.interfaces.mcp.tools import Tool, ToolRunning

MOST_TEXT = 100_000
LEFT_OUT = f"with it, this answer's text would pass {MOST_TEXT:,} characters: get it alone, or some of its lines"
GET = (
    "Read what a search found: documents as indexed, whole or some of their lines, such as a hit's first_line and "
    f"last_line. Read the best few at once, in one call. Texts past {MOST_TEXT:,} characters in all are left out, "
    "each saying so (`left_out`); with `fields`, each document's front matter too."
)


def _shown(document: DocumentText, room: int) -> ToolAnswer:
    shown: ToolAnswer = {"id": document.id, "title": document.title}
    if document.fields is not None:
        shown["fields"] = encoded(document.fields)
    if len(document.text) > room:
        return {**shown, "left_out": LEFT_OUT}
    return {**shown, "text": document.text}


def _get(running: ToolRunning, arguments: Mapping[str, object]) -> Json:
    documents = answered(running.read("get", cast("Json", dict(arguments))), Documents)
    shown, used = list[Json](), 0
    for document in documents.found:
        one = _shown(document, MOST_TEXT - used)
        used += len(document.text) if "text" in one else 0
        shown.append(one)
    answer: ToolAnswer = {"documents": shown}
    if documents.not_read:
        answer["not_read"] = [said(each) for each in documents.not_read]
    return answer


GET_TOOL = Tool("get", "Get", GET, schema_of(GetDocuments), _get, read_only=True)
