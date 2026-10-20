"""MCP server exposing search, vsearch, query, get and status."""

import asyncio
import logging
import sqlite3
import sys
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import embed, remote
from .search import Index

INSTRUCTIONS = """Search the user's local documents and notes, indexed by sift.
When a question touches what these collections cover, search here before answering from memory or searching the web,
and cite what you find; say so when nothing relevant turns up.

{collections}

- `query`: hybrid search with reranking. Best default for questions; slower than the others.
- `search`: exact keyword search (BM25). Best for names, identifiers, exact terms and "quoted phrases".
- `vsearch`: semantic search. Best when the wording of the answer is unknown.
Results carry `id` (collection/path), `lines`, `title`, `score` and chunk `text`. `get` returns a document, or
just some of its `lines`, such as a result's.
All search tools accept `collection`, a `path` glob such as "2024/*", and `filters` on front-matter fields: a value,
e.g. {{"categories": "cs.RO"}}, which a list field matches when it contains it, or a range, e.g.
{{"created": {{">=": "2024-06", "<": "2025"}}}}; dates compare at the precision given, so "<=": "2024" includes all
of 2024. `fields` adds front-matter fields to each result, e.g. ["url", "created"] (["*"] for all), and
`compact: true` leaves out the text: survey many results compactly, then read the ones that matter."""


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


def _run(fn, *args):
    """Call an Index method, turning expected failures into ToolError, the only kind whose message reaches the
    agent; anything else counts as a crash, and the MCP library tells the agent just "Error executing tool"."""
    try:
        return fn(*args)
    except (LookupError, ValueError, ConnectionError, sqlite3.Error) as e:  # ConnectionError: the embedding remote
        raise ToolError(str(e)) from None


def build(index=None):
    # Tools are async, so they run one at a time on the event-loop thread. That serialises GPU and SQLite work, and
    # keeps PyTorch out of worker threads, where it crashes on Python 3.15 (Windows, macOS).
    index = index or Index()
    mcp = MCPServer("sift", instructions=INSTRUCTIONS.format(collections=_collections(index)))

    @mcp.tool()
    async def search(
        query: str,
        k: int = 10,
        collection: str | None = None,
        path: str | None = None,
        filters: dict[str, Any] | None = None,
        fields: list[str] | None = None,
        compact: bool = False,
    ) -> list[dict]:
        """Exact keyword search ranked by BM25. Supports "phrases", prefix*, AND, OR, NOT. When no chunk holds
        all terms of a query without operators, chunks holding any of them are returned."""
        return _run(index.search, query, k, collection, path, filters, fields, compact)

    @mcp.tool()
    async def vsearch(
        query: str,
        k: int = 10,
        collection: str | None = None,
        path: str | None = None,
        filters: dict[str, Any] | None = None,
        fields: list[str] | None = None,
        compact: bool = False,
    ) -> list[dict]:
        """Semantic search: chunks closest in meaning to the query."""
        return _run(index.vsearch, query, k, collection, path, filters, fields, compact)

    @mcp.tool()
    async def query(
        query: str,
        k: int = 10,
        collection: str | None = None,
        path: str | None = None,
        filters: dict[str, Any] | None = None,
        fields: list[str] | None = None,
        compact: bool = False,
    ) -> list[dict]:
        """Hybrid search for RAG: keyword and semantic candidates, reranked by a cross-encoder."""
        return _run(index.query, query, k, collection, path, filters, fields, compact)

    @mcp.tool()
    async def get(id: str, lines: str | None = None) -> str:
        """Text of a document, by the `id` shown in search results: all of it, or `lines` such as "17-26".
        The id may leave out folders if the file name is unique, e.g. papers/2401.02225.md."""
        return _run(index.get, id, lines)

    @mcp.tool()
    async def status() -> dict:
        """Collections, document and chunk counts, and the embedding model."""
        return _run(index.status)

    return mcp, index


IDLE_CHECK_SECONDS = 30  # how often the server checks whether models and vectors can leave the GPU
SHUTDOWN_SECONDS = 3  # how long Ctrl+C waits for open requests before the server stops anyway


async def _release_gpu_when_idle():
    while True:
        await anyio.sleep(IDLE_CHECK_SECONDS)
        embed.release_idle()


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


async def _serve(mcp, host, port, stdio):
    # One event loop for the transport and the idle check: both stay on the main thread, where PyTorch must run.
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(_release_gpu_when_idle)
        if stdio:
            await mcp.run_stdio_async()
        else:
            print(f"sift MCP server on http://{host}:{port}/mcp", file=sys.stderr)
            await _serve_http(mcp, host, port)
        tasks.cancel_scope.cancel()


def serve(host="127.0.0.1", port=8765, stdio=False):
    mcp, index = build()
    print("loading models and vectors into RAM ...", file=sys.stderr)  # they move to the GPU when used
    if remote.REMOTE:  # the worker embeds and reranks; keyword search works without it
        try:
            remote.model()
        except ConnectionError as e:
            print(f"warning: {e}", file=sys.stderr)
    else:
        embed.embedder()
        embed.reranker()
    try:
        index._load()
    except LookupError as e:
        print(f"warning: {e}", file=sys.stderr)
    try:
        anyio.run(_serve, mcp, host, port, stdio)
    except BaseExceptionGroup as group:  # from the task group; the error inside is the transport's (or Ctrl+C)
        raise group.exceptions[0] from None
