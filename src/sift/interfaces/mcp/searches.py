import dataclasses
import difflib
from collections.abc import Generator, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import MappingProxyType
from typing import cast

from sift.api.errors import InvalidRequestError
from sift.api.links import FindMentions, FindSimilar, Links, ShowLinks
from sift.api.search import MOST_HITS, Options, SearchHybrid, SearchKeywords, SearchVectors
from sift.api.wire import Json, decoded, encoded
from sift.interfaces.acts.acting import answered
from sift.interfaces.mcp.answers import MOST_TEXT, ToolAnswer, compacted, hits_of, with_notes
from sift.interfaces.mcp.schemas import Schema, schema_of
from sift.interfaces.mcp.tools import Tool, ToolRunning

MOST_QUERIES = 20
MODES: Mapping[str, tuple[str, type[Options]]] = MappingProxyType(
    {"hybrid": ("query", SearchHybrid), "keyword": ("search", SearchKeywords), "semantic": ("vsearch", SearchVectors)}
)
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
    "prefix*, AND, OR, NOT (in capitals) and (parentheses). Keywords side by side must all be in a passage; if no "
    "passage holds them all, any will do, and the answer says `loosened`. `a NOT b c` leaves out what holds both b "
    f"and c: write `a NOT (b OR c)` for either. Several phrasings at once as a list of up to {MOST_QUERIES}, "
    f"searched alike and at the same time, each within the budget, answered in turn; k times their number is at "
    f"most {MOST_HITS:,}"
)
MODE = (
    "hybrid (default): keywords and meaning, reranked, for questions and topics; keyword: exact strings, such as "
    'names, versions "3.12.4", error codes, identifiers, DOIs, "quoted phrases"; semantic: meaning alone, for an '
    "idea whose wording you can't guess. The answer's `scored_by` says what its scores are: reranker (hybrid; higher "
    "is closer), keywords (BM25, from 0 up: more and rarer words found score higher) or vectors (cosine, at most 1). "
    "Hybrid falls back to keywords where a collection is not embedded"
)
FILTERS = (
    'only documents whose front matter passes every filter, each {"field", "operator", "values"} with an operator '
    "of =, !=, <, <=, >, >=. = with several values: any of them; != : none of them, and documents without the field "
    'pass. A list field matches when it holds a value; 2024 and "2024" match alike; dates compare at the precision '
    'given ("<=" "2024" covers all of 2024). E.g. [{"field": "year", "operator": ">=", "values": [2024]}, '
    '{"field": "tags", "operator": "=", "values": ["ml", "nlp"]}]'
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


def _refuse_too_many_hits(rest: dict[str, object], queries: tuple[str, ...], request: type[Options]) -> None:
    asked, n = decoded(request, {**rest, "query": queries[0]}), len(queries)
    if asked.k * n > MOST_HITS:
        said = f"{n} queries of {asked.k:,} hits each ask for more than {MOST_HITS:,} hits in all"
        raise InvalidRequestError(said, f"ask for {MOST_HITS // n:,} hits each at most, or give fewer queries", "k")


@contextmanager
def _named(field: str) -> Generator[None]:
    try:
        yield
    except InvalidRequestError as error:
        if error.field != "query" or field == "query":
            raise
        said = f"{field}{error.message.removeprefix('query')}" if error.message.startswith("query ") else error.message
        raise InvalidRequestError(said, error.remedy, field, code=error.code) from None


def _search(running: ToolRunning, arguments: Mapping[str, object]) -> Json:
    rest = dict(arguments)
    if "query" not in rest:
        raise InvalidRequestError("the search needs query", "give query, e.g. what the documents are about", "query")
    queries, mode = _queries(rest.pop("query")), rest.pop("mode", "hybrid")
    if not isinstance(mode, str) or mode not in MODES:
        raise _refused("mode", mode, _either(MODES))
    act, request = MODES[mode]
    _refuse_untaken("search", ("query", "mode", *_own(request, "query")), rest)
    alone = isinstance(arguments["query"], str)
    names = ["query" if alone else f"query[{at}]" for at in range(len(queries))]
    for name, query in zip(names, queries, strict=True):
        with _named(name):
            decoded(request, {**rest, "query": query})
    _refuse_too_many_hits(rest, queries, request)
    found, room = list[ToolAnswer](), MOST_TEXT
    with ThreadPoolExecutor(len(queries), thread_name_prefix="sift-query") as pool:
        asked = [pool.submit(running.read, act, cast("Json", {**rest, "query": query})) for query in queries]
        for name, answer in zip(names, asked, strict=True):
            with _named(name):
                shown, held = hits_of(answer.result(), room)
            room -= held
            found.append(shown)
    if alone:
        return cast("Json", found[0])
    return {"results": [{"query": query, **each} for query, each in zip(queries, found, strict=True)]}


def _links(answer: object) -> ToolAnswer:
    links = answered(answer, Links)
    shown = compacted(cast("dict[str, Json]", encoded(links)))
    _ = shown.pop("unsearched", None)
    return with_notes(shown, links.unsearched)


def _own(request: type, *but: str) -> tuple[str, ...]:
    return tuple(field.name for field in dataclasses.fields(request) if field.name not in but)


def _refuse_untaken(tool: str, taken: tuple[str, ...], given: Mapping[str, object]) -> None:
    for key in given:
        if key not in taken:
            near = difflib.get_close_matches(key, taken, 1)
            remedy = f"give {near[0]} instead" if near else f"leave out {key!r}: it takes {', '.join(taken)}"
            raise InvalidRequestError(f"{tool} takes no {key!r}", remedy, key)


def _related(running: ToolRunning, arguments: Mapping[str, object]) -> Json:
    rest = dict(arguments)
    kind = rest.pop("kind", "similar")
    if not isinstance(kind, str) or kind not in KINDS:
        raise _refused("kind", kind, _either(KINDS))
    act, request, direction = KINDS[kind]
    _refuse_untaken(f"related of kind {kind}", ("id", "kind", *_own(request, "id", "direction")), rest)
    if direction is None:
        return cast("Json", hits_of(running.read(act, cast("Json", rest)))[0])
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
        **_filtering(given),
    }
    return {**schema, "properties": properties}


def _filtering(properties: Mapping[str, Json]) -> Schema:
    filters = cast("Schema", properties["filters"])
    return {**properties, "filters": {**filters, "description": FILTERS}}


def _related_schema() -> Schema:
    mentions, links = _properties(schema_of(FindMentions)), _properties(schema_of(ShowLinks))
    given = _filtering({name: each for name, each in mentions.items() if name not in ("id", "k")})
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
