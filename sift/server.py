"""`sift serve`: the MCP server agents search through (search, vsearch, query, similar, get, status, links,
unresolved, mentions and reindex), with the UI for people under /ui (see `ui`)."""

import asyncio
import logging
import sqlite3
from typing import Annotated, Any, Literal

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from . import console
from . import models as _models
from .search import Index

log = logging.getLogger(__name__)

INSTRUCTIONS = """Search the user's local documents and notes, indexed by sift.
When a question touches what these collections cover, search here before answering from memory or searching the web,
and cite what you find; say so when nothing relevant turns up.

{collections}

- `query`: hybrid search with reranking. Best default for questions; slower than the others.
- `search`: exact keyword search (BM25). Best for names, identifiers, exact terms and "quoted phrases".
- `vsearch`: semantic search. Best when the wording of the answer is unknown.
- `similar`: documents like a given one, e.g. more papers like one found.
- `get`: a document, or some of its lines, such as a result's.
- `links`: the links of a note ([[wikilinks]], embeds, Markdown links): those it holds, and those into it (backlinks).
- `unresolved`: link targets no note answers to yet: the notes a collection lacks.
- `mentions`: documents that name a note without linking to it.
- `reindex`: after writing, changing or deleting files in a collection, index them again, so searches find them.
Results carry `id` (collection/path), `lines`, `title`, `score` and the passage's `text`. Scores rank the results of
one call; each tool has its own scale, so don't compare them across tools. All search tools take `collection`, a
`path` glob, `filters` on front-matter fields, `fields` to show, and `links_to` and `linked_from` to keep to a note's
links; with `compact: true` results leave out the text: survey many results compactly, then read the ones that
matter."""


def _collections(index):
    """The indexed collections with their folders, sizes and front-matter fields, read when the server starts."""
    rows = index.db.execute("""SELECT co.id, co.name, co.path, count(d.id) FROM collections co
                               LEFT JOIN docs d ON d.collection = co.id GROUP BY co.id ORDER BY co.name""").fetchall()
    if not rows:
        return "No collections are indexed yet."
    lines, (linked, broken) = [], index.link_counts()
    for cid, name, folder, n in rows:
        line = f"- {name} ({folder}): {n:,} document{'' if n == 1 else 's'}"
        if fields := index.fields(cid, n):
            line += "; front matter: " + ", ".join(fields)
        if cid in linked:
            line += f"; {console.count(linked[cid], 'link')}, {broken[cid]:,} unresolved"
        lines.append(line)
    return "Indexed collections:\n" + "\n".join(lines)


def _run(fn, *args, **kwargs):
    """Call an Index method. Any failure becomes a ToolError, the only kind whose message the MCP library passes on
    (it tells the agent just "Error executing tool" otherwise): expected ones as their message, others, which are
    bugs, with their type, and with a traceback in the server's log."""
    try:
        return fn(*args, **kwargs)
    except (LookupError, ValueError, ConnectionError, sqlite3.Error) as e:  # ConnectionError: the embedding remote
        raise ToolError(str(e)) from None
    except Exception as e:
        log.exception("tool failed")
        raise ToolError(f"{type(e).__name__}: {e}") from None


# What agents see of each parameter: they read the tools' schemas, not sift's code.
Id = Annotated[
    str,
    Field(
        description='A document\'s id from a search result, e.g. "papers/2024/attention.md"; folders may be left '
        'out if the file name is unique: "papers/attention.md".'
    ),
]
Note = Annotated[
    str,
    Field(
        description='A note\'s id, e.g. "notes/People/Hans Müller.md", or its name as links write it, e.g. "Hans '
        'Müller", if one note answers to it.'
    ),
]
K = Annotated[int, Field(description="Number of results.")]
Collection = Annotated[str | None, Field(description="Search only this collection.")]
PathGlob = Annotated[
    str | None, Field(description='Only documents whose path in the collection matches this glob, e.g. "2024/*".')
]
Filters = Annotated[
    dict[str, Any] | None,
    Field(
        description='Conditions on front-matter fields, all of which must hold: a value, e.g. {"categories": '
        '"cs.RO"}, which a list field matches if it holds it; a list for any of its values, e.g. {"categories": '
        '["cs.RO", "cs.CV"]}; {"not": a value or list} for none of them (documents without the field pass); or a '
        'range, e.g. {"created": {">=": "2024-06", "<": "2025"}}. Dates compare at the precision given, so '
        '{"<=": "2024"} includes all of 2024. A wikilink, e.g. {"org": "[[Acme AG]]"}, matches a link to that note '
        "however it is written."
    ),
]
LinksTo = Annotated[
    list[str] | None,
    Field(
        description="Only documents that link to any of these notes, named by id or as links name them, e.g. "
        '["Acme AG"] or ["[[Acme AG]]"]: search a note\'s backlinks. A name is resolved within `collection`, if '
        "given; one that several notes answer to is an error that lists them."
    ),
]
LinkedFrom = Annotated[
    list[str] | None,
    Field(description='Only documents that one of these notes links to, e.g. ["notes/People/Hans Müller.md"].'),
]
Fields = Annotated[
    list[str] | None,
    Field(description='Front-matter fields to add to each result, e.g. ["url", "created"]; ["*"] for all of them.'),
]
Compact = Annotated[bool, Field(description="Leave out the text: survey many results, then `get` those that matter.")]
AllPassages = Annotated[
    bool,
    Field(description="Every matching passage, several per document; by default only each document's best one."),
]
Sort = Annotated[
    str | None,
    Field(
        description='Order the results by a front-matter field instead of by score, with "-" first for descending, '
        'e.g. "-created" for the newest first; results without the field come last.'
    ),
]
READ_ONLY = ToolAnnotations(read_only_hint=True)


def _add_search(mcp, index, name, description, query_description):
    """Add the tool for the Index search called `name`."""
    method = getattr(index, name)

    async def tool(
        query: Annotated[str, Field(description=query_description)],
        k: K = 10,
        collection: Collection = None,
        path: PathGlob = None,
        filters: Filters = None,
        fields: Fields = None,
        compact: Compact = False,
        all_passages: AllPassages = False,
        sort: Sort = None,
        links_to: LinksTo = None,
        linked_from: LinkedFrom = None,
    ) -> list[dict]:
        return _run(
            method,
            query,
            k,
            collection=collection,
            path=path,
            filters=filters,
            fields=fields,
            compact=compact,
            all_passages=all_passages,
            sort=sort,
            links_to=links_to,
            linked_from=linked_from,
        )

    mcp.tool(name, description=description, annotations=READ_ONLY)(tool)


def build(index=None):
    # Tools are async, so they run one at a time on the event-loop thread. That serialises GPU and SQLite work, and
    # keeps PyTorch out of worker threads, where it crashes on Python 3.15 (Windows, macOS).
    index = index or Index()
    mcp = MCPServer("sift", instructions=INSTRUCTIONS.format(collections=_collections(index)))
    _add_search(
        mcp,
        index,
        "search",
        "Exact keyword search, ranked by BM25 (scores from 0 up, higher is better), over each passage's text, its "
        "document's title and front-matter values, such as authors and ids. When no passage holds all terms of a "
        "query without operators, passages holding any of them are returned; a query with operators is taken as it "
        "is, so drop terms when it finds nothing.",
        'Keywords. "Exact phrases", prefix*, AND, OR, NOT (in capitals) and parentheses work, e.g. '
        '"notice period" AND (contract OR agreement); title: and meta: limit a term to the title or the front matter.',
    )
    _add_search(
        mcp,
        index,
        "vsearch",
        "Semantic search: passages closest in meaning to the query (cosine similarity, -1 to 1).",
        "What to find, in any wording.",
    )
    _add_search(
        mcp,
        index,
        "query",
        "Hybrid search for RAG: keyword and semantic candidates, reranked by a cross-encoder. Best for questions. "
        "Scores are the reranker's raw scores, higher for more relevant; with the default reranker, close matches "
        "score about 2.5 to 4, and unrelated results below 2.",
        "A question, or a description of what to find.",
    )

    @mcp.tool(annotations=READ_ONLY)
    async def similar(
        id: Id,
        k: K = 10,
        collection: Collection = None,
        path: PathGlob = None,
        filters: Filters = None,
        fields: Fields = None,
        compact: Compact = False,
        all_passages: AllPassages = False,
        sort: Sort = None,
        links_to: LinksTo = None,
        linked_from: LinkedFrom = None,
    ) -> list[dict]:
        """Documents closest in meaning to a document, e.g. more papers like one found; the document itself is left
        out (cosine similarity, -1 to 1)."""
        return _run(
            index.similar,
            id,
            k,
            collection=collection,
            path=path,
            filters=filters,
            fields=fields,
            compact=compact,
            all_passages=all_passages,
            sort=sort,
            links_to=links_to,
            linked_from=linked_from,
        )

    @mcp.tool(annotations=READ_ONLY)
    async def get(
        id: Id,
        lines: Annotated[
            str | int | None,
            Field(description='Only these lines, e.g. a result\'s "17-26", or one line number; all if left out.'),
        ] = None,
        fields: Fields = None,
    ) -> str | dict:
        """Text of a document, as indexed: all of it, or some of its lines. With `fields`, the document as
        {id, title, fields, text}."""
        if fields:
            return _run(index.document, id, lines, fields)
        return _run(index.get, id, lines)

    @mcp.tool(annotations=READ_ONLY)
    async def status() -> dict:
        """Collections with their folders (`path`), document and chunk counts, link counts, and the embedding model."""
        return _run(index.status)

    @mcp.tool(annotations=READ_ONLY)
    async def links(
        id: Note,
        direction: Annotated[
            Literal["out", "in", "both"],
            Field(description='"out": the links the note holds; "in": the documents linking it (backlinks).'),
        ] = "out",
        depth: Annotated[
            Literal[1, 2],
            Field(description="2: also the notes one link further, each with the note it was reached `through`."),
        ] = 1,
        field: Annotated[
            str | None,
            Field(
                description='Only links in this front-matter field, e.g. "org": with direction "in", the notes whose '
                "org links to this one."
            ),
        ] = None,
        k: Annotated[int, Field(description="At most this many links out, and this many documents in.")] = 50,
    ) -> dict:
        """Links between Markdown notes: [[wikilinks]], ![[embeds]] and [Markdown](links.md), in the text and in
        front-matter values. "out" lists each link the note holds, with its line and front-matter field, and where it
        leads: the document's id, and `via` path, name or alias; or `ambiguous`, with the candidates; or `unresolved`,
        a note not written yet. "in" lists the documents linking the note, each with the lines that do, so you see
        the context without `get`. Links resolve within their collection, as Obsidian resolves them."""
        return _run(index.links, id, direction, depth, field, k)

    @mcp.tool(annotations=READ_ONLY)
    async def unresolved(
        collection: Annotated[str | None, Field(description="Only this collection's links.")] = None,
        k: Annotated[int, Field(description="Number of targets.")] = 50,
    ) -> list[dict]:
        """Link targets no document answers to: the notes a collection lacks, most linked first, each with how many
        documents link it and the ids of a few of them."""
        return _run(index.unresolved, collection, k)

    @mcp.tool(annotations=READ_ONLY)
    async def mentions(
        id: Note,
        k: K = 10,
        collection: Collection = None,
        also: Annotated[
            list[str] | None,
            Field(description='Front-matter fields of the note whose values count as its names too, e.g. ["emails"].'),
        ] = None,
        fields: Fields = None,
        compact: Compact = False,
    ) -> list[dict]:
        """Documents that name a note without linking to it (unlinked mentions), in any collection: found by keyword
        search for the note's names as phrases, its file name, title and aliases. Results look like search results;
        the note itself and the documents linking it are left out. Read the passages to judge which mean the note."""
        return _run(index.mentions, id, k, collection=collection, also=also, fields=fields, compact=compact)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True))
    async def reindex(
        paths: Annotated[
            list[str],
            Field(
                description='Files to index again: ids, e.g. "notes/People/Hans Müller.md", or absolute paths inside '
                "a collection. A file that is gone has its documents removed."
            ),
        ],
    ) -> list[dict]:
        """Index files again right after writing, changing or deleting them, so that the next search finds them as
        they are now: their text, front matter and links, and, if the index is embedded, their passages for vsearch
        and query. Changes only sift's index, never a file. Per file: its id and status (added, changed, unchanged,
        removed, not found, skipped, unreadable), its passages (`chunks`) and how many are `embedded`."""
        return _run(index.reindex, paths)

    return mcp, index


IDLE_CHECK_SECONDS = 30  # how often the server checks whether models and vectors can leave the GPU
SHUTDOWN_SECONDS = 3  # how long Ctrl+C waits for open requests before the server stops anyway


async def release_when_idle(models):
    """Every IDLE_CHECK_SECONDS, let the models hand the GPU back if it has been idle long enough."""
    while True:
        await anyio.sleep(IDLE_CHECK_SECONDS)
        models.release_idle()


def http_app(mcp, index, host):
    """The app `sift serve` runs over HTTP: MCP under /mcp, and the UI under /ui, where the address alone leads."""
    from starlette.responses import RedirectResponse
    from starlette.routing import Mount, Route

    from . import ui

    async def home(request):
        return RedirectResponse(ui.PREFIX)

    app = mcp.streamable_http_app(host=host)  # the host decides on DNS-rebinding protection
    app.router.routes += [Mount("/ui", ui.app(index, host)), Route("/", home)]
    return app


async def _serve_http(mcp, index, host, port):
    await serve_app(http_app(mcp, index, host), host, port, mcp.settings.log_level.lower())


async def serve_app(app, host, port, log_level="info", access_log=True, **options):
    """Serve an HTTP app with uvicorn, but with a time limit on shutdown. Without one, uvicorn waits for every
    connection to close (Python 3.12+), even after a second Ctrl+C, and a client that leaves a request unfinished
    keeps the server from ever stopping. Options go to uvicorn, e.g. its TLS files."""
    import uvicorn

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=access_log,
        timeout_graceful_shutdown=SHUTDOWN_SECONDS,
        **options,
    )
    logging.getLogger("uvicorn.error").addFilter(_no_cancelled_requests)  # after Config, which sets up logging
    await uvicorn.Server(config).serve()


def _no_cancelled_requests(record):
    """Drop the traceback uvicorn logs for each request it cancels at shutdown; it says beforehand that it does."""
    return not (record.exc_info and isinstance(record.exc_info[1], asyncio.CancelledError))


async def _serve(mcp, index, host, port, stdio, models):
    # One event loop for the transport and the idle check: both stay on the main thread, where PyTorch must run.
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(release_when_idle, models)
        if stdio:
            await mcp.run_stdio_async()
        else:
            from .ui import LOCAL_HOSTS, PREFIX, address

            listening = f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"
            console.say(f"sift MCP server on {listening}/mcp; browse the index at {address(host, port)}{PREFIX}")
            if host not in LOCAL_HOSTS:
                console.say(
                    f"warning: listening on {host}, so anyone who reaches port {port} can read every indexed "
                    "document, through MCP and the UI: sift has no authentication"
                )
            await _serve_http(mcp, index, host, port)
        tasks.cancel_scope.cancel()


def serve(host="127.0.0.1", port=8765, stdio=False, models=None):
    """Run the MCP server, with `models` (default: `models.default()`) loaded up front, so the first search is quick."""
    models = models or _models.default()
    mcp, index = build(Index(models=models))
    console.say("loading models and vectors into RAM ...")  # they move to the GPU when used
    for step in (models.load, index.load):
        try:
            step()
        except (ConnectionError, LookupError) as e:  # a worker that doesn't answer, no vectors yet: keywords work
            console.say(f"warning: {e}")
    try:
        anyio.run(_serve, mcp, index, host, port, stdio, models)
    except BaseExceptionGroup as group:  # from the task group; the error inside is the transport's (or Ctrl+C)
        raise group.exceptions[0] from None
