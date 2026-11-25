import dataclasses
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import cast

from sift.api.errors import InvalidRequestError
from sift.api.links import FindMentions, FindSimilar, Links, ShowLinks
from sift.api.search import SearchHybrid
from sift.api.wire import Json, encoded
from sift.interfaces.acts.acting import answered
from sift.interfaces.mcp.answers import ToolAnswer, compacted, hits_of, with_notes
from sift.interfaces.mcp.schemas import Schema, schema_of
from sift.interfaces.mcp.tools import Tool, ToolRunning

MOST_QUERIES = 20
MODES: Mapping[str, str] = MappingProxyType({"hybrid": "query", "keyword": "search", "semantic": "vsearch"})
KINDS: Mapping[str, tuple[str, type, str | None]] = MappingProxyType(
    {
        "similar": ("similar", FindSimilar, None),
        "links": ("links", ShowLinks, "out"),
        "backlinks": ("links", ShowLinks, "in"),
        "mentions": ("mentions", FindMentions, None),
    }
)
QUERY = (
    'what to find: a question or a description (hybrid, semantic), or keywords (keyword): words, "phrases", '
    f"prefix*, AND, OR, NOT and (parentheses); several phrasings at once as a list of up to {MOST_QUERIES}, each "
    "searched alike and answered in turn"
)
MODE = (
    "hybrid (default): keywords and meaning, reranked, for questions and topics; keyword: exact strings, such as "
    'names, versions "3.12.4", error codes, identifiers, DOIs, "quoted phrases"; semantic: meaning alone, for an '
    "idea whose wording you can't guess"
)
KIND = (
    "similar (default): documents closest in meaning to it; links: the links it holds and where each leads; "
    "backlinks: the documents linking it, with the lines that do; mentions: documents naming it without a link"
)
RELATED_ID = (
    "the document or note: its id from a search result (collection/path), its absolute path, or a note's name as "
    'links write it, e.g. "Acme AG"'
)
RELATED_K = "how many: results of similar and mentions (default 10), links and linking documents (default 50)"
SEARCH = (
    "Search the user's indexed documents and notes. Hybrid (the default) for questions and topics, keyword for exact "
    "strings, semantic for ideas. Each hit: its id (collection/path), title, the lines it spans, a score that ranks "
    "this call's hits alone, and the passage's text. To survey many, compact: true leaves the text out; then `get` "
    "the best few by id and lines, in one call. One phrasing rarely finds everything: give several as a list, and "
    "search again with the words the hits use. `more`: further hits wait; page on with offset."
)
RELATED = (
    "From a document or note you have, what relates to it: documents alike in meaning (to widen a search from its "
    "best hit), its links, its backlinks, or the documents that name it without linking it. Links resolve within "
    "their collection, as Obsidian resolves them."
)


def _either(names: Iterable[str]) -> str:
    *first, last = names
    return f"{', '.join(first)} or {last}" if first else last


def _refused(field: str, given: object, wanted: str) -> InvalidRequestError:
    return InvalidRequestError(f"{field} is {given!r}, not {wanted}", f"give {field} as {wanted}", field)


def _queries(given: object) -> tuple[str, ...]:
    if isinstance(given, str):
        return (given,)
    listed = cast("list[object]", given) if isinstance(given, list) else None
    if listed is None or not all(isinstance(each, str) for each in listed):
        raise _refused("query", cast("object", given), "a text or a list of texts")
    if not 1 <= len(listed) <= MOST_QUERIES:
        raise _refused("query", cast("object", given), f"1 to {MOST_QUERIES} texts")
    return tuple(cast("list[str]", listed))


def _search(running: ToolRunning, arguments: Mapping[str, object]) -> Json:
    rest = dict(arguments)
    if "query" not in rest:
        raise InvalidRequestError("the search needs query", "give query, e.g. what the documents are about", "query")
    queries, mode = _queries(rest.pop("query")), rest.pop("mode", "hybrid")
    if not isinstance(mode, str) or mode not in MODES:
        raise _refused("mode", mode, _either(MODES))
    found = [hits_of(running.read(MODES[mode], cast("Json", {**rest, "query": query}))) for query in queries]
    if isinstance(arguments["query"], str):
        return cast("Json", found[0])
    return {"results": [{"query": query, **each} for query, each in zip(queries, found, strict=True)]}


def _links(answer: object) -> ToolAnswer:
    links = answered(answer, Links)
    shown = cast("ToolAnswer", compacted(encoded(links)))
    _ = shown.pop("unsearched", None)
    return with_notes(shown, links.unsearched)


def _refuse_untaken(kind: str, request: type, given: Mapping[str, object]) -> None:
    own = (field.name for field in dataclasses.fields(request) if field.name not in ("id", "direction"))
    taken = ("id", "kind", *own)
    for key in given:
        if key not in taken:
            leave = f"leave out {key!r}: it takes {', '.join(taken)}"
            raise InvalidRequestError(f"related of kind {kind} takes no {key!r}", leave, key)


def _related(running: ToolRunning, arguments: Mapping[str, object]) -> Json:
    rest = dict(arguments)
    kind = rest.pop("kind", "similar")
    if not isinstance(kind, str) or kind not in KINDS:
        raise _refused("kind", kind, _either(KINDS))
    act, request, direction = KINDS[kind]
    _refuse_untaken(kind, request, rest)
    if direction is None:
        return cast("Json", hits_of(running.read(act, cast("Json", rest))))
    return cast("Json", _links(running.read(act, cast("Json", {**rest, "direction": direction}))))


def _properties(schema: Schema) -> Schema:
    return cast("Schema", schema["properties"])


def _search_schema() -> Schema:
    schema = schema_of(SearchHybrid)
    texts: Schema = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": MOST_QUERIES}
    given = {name: each for name, each in _properties(schema).items() if name != "query"}
    properties: Schema = {
        "query": {"anyOf": [{"type": "string"}, texts], "description": QUERY},
        "mode": {"type": "string", "enum": list[Json](MODES), "default": "hybrid", "description": MODE},
        **given,
    }
    return {**schema, "properties": properties}


def _related_schema() -> Schema:
    mentions, links = _properties(schema_of(FindMentions)), _properties(schema_of(ShowLinks))
    given = {name: each for name, each in mentions.items() if name not in ("id", "k")}
    properties: Schema = {
        "id": {"type": "string", "description": RELATED_ID},
        "kind": {"type": "string", "enum": list[Json](KINDS), "default": "similar", "description": KIND},
        "k": {"type": "integer", "minimum": 1, "description": RELATED_K},
        **given,
        "depth": links["depth"],
        "field": links["field"],
    }
    return {"type": "object", "properties": properties, "required": ["id"], "additionalProperties": False}


SEARCH_TOOL = Tool("search", "Search", SEARCH, _search_schema(), _search, read_only=True)
RELATED_TOOL = Tool("related", "Related", RELATED, _related_schema(), _related, read_only=True)
