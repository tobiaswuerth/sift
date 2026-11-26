from collections.abc import Mapping
from typing import cast

from sift.api.documents import ID, Documents, DocumentText, GetDocuments
from sift.api.errors import InvalidRequestError
from sift.api.requests import shown_value
from sift.api.wire import Json, encoded
from sift.domain.text.lines import NEWLINE, line_count
from sift.interfaces.acts.acting import answered
from sift.interfaces.mcp.answers import FULL, MOST_TEXT, ToolAnswer, said
from sift.interfaces.mcp.schemas import Schema, schema_of
from sift.interfaces.mcp.tools import Tool, ToolRunning

REST = f"lines {{first:,}} to {{last:,}} are left out, as {FULL}: get them with first_line {{first}}"
LATER = f"left out, as {FULL}: get it in a call of its own"
LONG_LINE = (
    f"line {{line:,}} goes on for {{more:,}} characters more than the {MOST_TEXT:,} an answer holds; they are left out"
)
AFTER = ", and the lines after it: get those with first_line {first}"
ONE = 'give each document as its id, or as {"id": "...", "first_line": 17, "last_line": 26} for some of its lines'
DOCUMENTS = "the documents to read: each its id, or an object with its id and the lines to read of it"
GET = (
    "Read what a search found: documents as indexed, whole or some of their lines, such as a hit's first_line and "
    f"last_line. Read the best few at once, in one call. An answer holds {MOST_TEXT:,} characters of text: a text "
    "that passes them is cut at a line and says which lines are left out and how to get them (`left_out`). With "
    "`fields`, each document's front matter too."
)


def _cut(document: DocumentText, room: int) -> ToolAnswer:
    text, first = document.text, document.first_line
    whole = text[: text.rfind(NEWLINE, 0, room + 1) + 1]
    if whole:
        last = first + whole.count(NEWLINE) - 1
        left = REST.format(first=last + 1, last=first + line_count(text) - 1)
        return {"text": whole, "first_line": first, "last_line": last, "lines": document.lines, "left_out": left}
    if room < MOST_TEXT:
        return {"left_out": LATER}
    line, _, after = text.partition(NEWLINE)
    left = LONG_LINE.format(line=first, more=len(line) - room) + (AFTER.format(first=first + 1) if after else "")
    return {"text": line[:room], "first_line": first, "last_line": first, "lines": document.lines, "left_out": left}


def _shown(document: DocumentText, room: int) -> ToolAnswer:
    shown: ToolAnswer = {"id": document.id, "title": document.title}
    if document.fields is not None:
        shown["fields"] = encoded(document.fields)
    return {**shown, **({"text": document.text} if len(document.text) <= room else _cut(document, room))}


def _wanted(given: object, at: int) -> object:
    if isinstance(given, str):
        return {"id": given}
    if not isinstance(given, dict):
        field = f"documents[{at}]"
        raise InvalidRequestError(f"{field} is {shown_value(given)}, neither an id nor an object", ONE, field)
    return cast("object", given)


def _get(running: ToolRunning, arguments: Mapping[str, object]) -> Json:
    asked, given = dict(arguments), arguments.get("documents")
    if isinstance(given, list):
        asked["documents"] = [_wanted(each, at) for at, each in enumerate(cast("list[object]", given))]
    documents = answered(running.read("get", cast("Json", asked)), Documents)
    shown, used = list[Json](), 0
    for document in documents.found:
        one = _shown(document, MOST_TEXT - used)
        used += len(cast("str", one.get("text", "")))
        shown.append(one)
    answer: ToolAnswer = {"documents": shown}
    if documents.not_read:
        answer["not_read"] = [said(each) for each in documents.not_read]
    return answer


def _get_schema() -> Schema:
    schema = schema_of(GetDocuments)
    properties = cast("Schema", schema["properties"])
    listed = cast("Schema", properties["documents"])
    either: Schema = {"anyOf": [{"type": "string", "description": ID}, listed["items"]]}
    return {**schema, "properties": {**properties, "documents": {**listed, "items": either, "description": DOCUMENTS}}}


GET_TOOL = Tool("get", "Get", GET, _get_schema(), _get, read_only=True)
