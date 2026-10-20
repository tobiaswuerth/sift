"""MCP server exposing search, vsearch, query, get and status."""

import asyncio
import logging
import sqlite3
from typing import Annotated, Any

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
- `get`: a document, or some of its lines, such as a result's.
Results carry `id` (collection/path), `lines`, `title`, `score` and the passage's `text`. Scores rank the results of
one call; each tool has its own scale, so don't compare them across tools. All search tools take `collection`, a
`path` glob, `filters` on front-matter fields and `fields` to show; with `compact: true` results leave out the text:
survey many results compactly, then read the ones that matter."""


def _collections(index):
    """The indexed collections with their sizes and front-matter fields, read when the server starts."""
    rows = index.db.execute("""SELECT co.id, co.name, count(d.id) FROM collections co
                               LEFT JOIN docs d ON d.collection = co.id GROUP BY co.id ORDER BY co.name""").fetchall()
    if not rows:
        return "No collections are indexed yet."
    lines = []
    for cid, name, n in rows:
        line = f"- {name}: {n:,} document{'' if n == 1 else 's'}"
        if fields := index.fields(cid, n):
            line += "; front matter: " + ", ".join(fields)
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
        '{"<=": "2024"} includes all of 2024.'
    ),
]
Fields = Annotated[
    list[str] | None,
    Field(description='Front-matter fields to add to each result, e.g. ["url", "created"]; ["*"] for all of them.'),
]
Compact = Annotated[bool, Field(description="Leave out the text: survey many results, then `get` those that matter.")]
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
    ) -> list[dict]:
        options = {"collection": collection, "path": path, "filters": filters, "fields": fields, "compact": compact}
        return _run(method, query, k, **options)

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
        "Exact keyword search, ranked by BM25 (scores from 0 up, higher is better). When no passage holds all "
        "terms of a query without operators, passages holding any of them are returned; a query with operators is "
        "taken as it is, so drop terms when it finds nothing.",
        'Keywords. "Exact phrases", prefix*, AND, OR, NOT (in capitals) and parentheses work, e.g. '
        '"stroke" AND (painting OR rendering).',
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
        "Hybrid search for RAG: keyword and semantic candidates, reranked by a cross-encoder. Best for questions.",
        "A question, or a description of what to find.",
    )

    @mcp.tool(annotations=READ_ONLY)
    async def get(
        id: Annotated[
            str,
            Field(
                description='A document\'s id from a search result, e.g. "papers/2024/attention.md"; folders may be '
                'left out if the file name is unique: "papers/attention.md".'
            ),
        ],
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
        """Collections, document and chunk counts, and the embedding model."""
        return _run(index.status)

    return mcp, index


IDLE_CHECK_SECONDS = 30  # how often the server checks whether models and vectors can leave the GPU
SHUTDOWN_SECONDS = 3  # how long Ctrl+C waits for open requests before the server stops anyway


async def release_when_idle(models):
    """Every IDLE_CHECK_SECONDS, let the models hand the GPU back if it has been idle long enough."""
    while True:
        await anyio.sleep(IDLE_CHECK_SECONDS)
        models.release_idle()


async def _serve_http(mcp, host, port):
    app = mcp.streamable_http_app(host=host)  # the host decides on DNS-rebinding protection
    await serve_app(app, host, port, mcp.settings.log_level.lower())


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


async def _serve(mcp, host, port, stdio, models):
    # One event loop for the transport and the idle check: both stay on the main thread, where PyTorch must run.
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(release_when_idle, models)
        if stdio:
            await mcp.run_stdio_async()
        else:
            console.say(f"sift MCP server on http://{host}:{port}/mcp")
            await _serve_http(mcp, host, port)
        tasks.cancel_scope.cancel()


def serve(host="127.0.0.1", port=8765, stdio=False, models=None):
    """Run the MCP server, with `models` (default: `models.default()`) loaded up front, so the first search is quick."""
    models = models or _models.default()
    mcp, index = build(Index(models=models))
    console.say("loading models and vectors into RAM ...")  # they move to the GPU when used
    for step in (models.load, index.load):
        try:
            step()
        except (
            ConnectionError,
            LookupError,
        ) as e:  # a worker that doesn't answer, no vectors yet: keyword search works
            console.say(f"warning: {e}")
    try:
        anyio.run(_serve, mcp, host, port, stdio, models)
    except BaseExceptionGroup as group:  # from the task group; the error inside is the transport's (or Ctrl+C)
        raise group.exceptions[0] from None
