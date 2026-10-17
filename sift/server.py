"""MCP server exposing search, vsearch, query, get and status."""

import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import embed
from .search import Index

INSTRUCTIONS = """Search the user's local text and Markdown files.
- `search`: exact keyword search (BM25). Best for names, identifiers, exact terms and "quoted phrases".
- `vsearch`: semantic search. Best when the wording of the answer is unknown.
- `query`: hybrid search with reranking. Best default for questions; slower than the others.
Results carry `id` (collection/path), `lines`, `title`, `score` and chunk `text`. Use `get` for a full document.
All search tools accept `collection`, a `path` glob such as "2024/*", and `filters` on front-matter fields,
e.g. {"categories": "cs.RO"}; a list field matches when it contains the value."""


def build(index=None):
    # Tools are async, so they run one at a time on the event-loop thread. That serialises GPU and SQLite work, and
    # keeps PyTorch out of worker threads, where it crashes on Python 3.15 (Windows, macOS).
    index = index or Index()
    mcp = MCPServer("sift", instructions=INSTRUCTIONS)

    @mcp.tool()
    async def search(
        query: str,
        k: int = 10,
        collection: str | None = None,
        path: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Exact keyword search ranked by BM25. Supports "phrases", prefix*, AND, OR, NOT."""
        return index.search(query, k, collection, path, filters)

    @mcp.tool()
    async def vsearch(
        query: str,
        k: int = 10,
        collection: str | None = None,
        path: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Semantic search: chunks closest in meaning to the query."""
        return index.vsearch(query, k, collection, path, filters)

    @mcp.tool()
    async def query(
        query: str,
        k: int = 10,
        collection: str | None = None,
        path: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Hybrid search for RAG: keyword and semantic candidates, reranked by a cross-encoder."""
        return index.query(query, k, collection, path, filters)

    @mcp.tool()
    async def get(id: str) -> str:
        """Full text of a document, by the `id` shown in search results."""
        return index.get(id)

    @mcp.tool()
    async def status() -> dict:
        """Collections, document and chunk counts, and the embedding model."""
        return index.status()

    return mcp, index


def serve(host="127.0.0.1", port=8765, stdio=False):
    mcp, index = build()
    print("loading models and vectors ...", file=sys.stderr)
    embed.embedder()
    embed.reranker()
    try:
        index._vectors()
    except LookupError as e:
        print(f"warning: {e}", file=sys.stderr)
    if stdio:
        mcp.run("stdio")
    else:
        print(f"sift MCP server on http://{host}:{port}/mcp", file=sys.stderr)
        mcp.run("streamable-http", host=host, port=port)
