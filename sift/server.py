"""MCP server exposing search, vsearch, query, get and status."""

import sqlite3
import sys
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import embed
from .search import Index

INSTRUCTIONS = """Search the user's local text and Markdown files, indexed by sift.
When a question touches what these collections cover, search here before answering from memory or searching the web,
and cite what you find; say so when nothing relevant turns up.

{collections}

- `query`: hybrid search with reranking. Best default for questions; slower than the others.
- `search`: exact keyword search (BM25). Best for names, identifiers, exact terms and "quoted phrases".
- `vsearch`: semantic search. Best when the wording of the answer is unknown.
Results carry `id` (collection/path), `lines`, `title`, `score` and chunk `text`. Use `get` for a full document.
All search tools accept `collection`, a `path` glob such as "2024/*", and `filters` on front-matter fields,
e.g. {{"categories": "cs.RO"}}; a list field matches when it contains the value."""


def _collections(index):
    """The indexed collections with their sizes, read when the server starts."""
    rows = index.db.execute("""SELECT co.name, count(d.id) FROM collections co LEFT JOIN docs d ON d.collection = co.id
                               GROUP BY co.id ORDER BY co.name""").fetchall()
    if not rows:
        return "No collections are indexed yet."
    lines = [f"- {name}: {n:,} document{'' if n == 1 else 's'}" for name, n in rows]
    return "Indexed collections:\n" + "\n".join(lines)


def _run(fn, *args):
    """Call an Index method, turning expected failures into ToolError, the only kind whose message reaches the
    agent; anything else counts as a crash, and the MCP library tells the agent just "Error executing tool"."""
    try:
        return fn(*args)
    except (LookupError, ValueError, sqlite3.Error) as e:
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
    ) -> list[dict]:
        """Exact keyword search ranked by BM25. Supports "phrases", prefix*, AND, OR, NOT."""
        return _run(index.search, query, k, collection, path, filters)

    @mcp.tool()
    async def vsearch(
        query: str,
        k: int = 10,
        collection: str | None = None,
        path: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Semantic search: chunks closest in meaning to the query."""
        return _run(index.vsearch, query, k, collection, path, filters)

    @mcp.tool()
    async def query(
        query: str,
        k: int = 10,
        collection: str | None = None,
        path: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Hybrid search for RAG: keyword and semantic candidates, reranked by a cross-encoder."""
        return _run(index.query, query, k, collection, path, filters)

    @mcp.tool()
    async def get(id: str) -> str:
        """Full text of a document, by the `id` shown in search results."""
        return _run(index.get, id)

    @mcp.tool()
    async def status() -> dict:
        """Collections, document and chunk counts, and the embedding model."""
        return _run(index.status)

    return mcp, index


IDLE_CHECK_SECONDS = 30  # how often the server checks whether models and vectors can leave the GPU


async def _release_gpu_when_idle():
    while True:
        await anyio.sleep(IDLE_CHECK_SECONDS)
        embed.release_idle()


async def _serve(mcp, host, port, stdio):
    # One event loop for the transport and the idle check: both stay on the main thread, where PyTorch must run.
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(_release_gpu_when_idle)
        if stdio:
            await mcp.run_stdio_async()
        else:
            print(f"sift MCP server on http://{host}:{port}/mcp", file=sys.stderr)
            await mcp.run_streamable_http_async(host=host, port=port)
        tasks.cancel_scope.cancel()


def serve(host="127.0.0.1", port=8765, stdio=False):
    mcp, index = build()
    print("loading models and vectors into RAM ...", file=sys.stderr)  # they move to the GPU when used
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
