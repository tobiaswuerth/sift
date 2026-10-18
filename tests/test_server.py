import asyncio
import json
import threading

import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer

from sift import server
from sift.search import Index


def call(mcp, name, args=None):
    async def go():
        async with Client(mcp) as client:
            return await client.call_tool(name, args or {})

    return asyncio.run(go())


def data(result):
    """A tool's return value, whether MCP sent it as structured content or as JSON text."""
    assert not result.is_error, result.content
    content = result.structured_content
    if content is None:
        return json.loads(result.content[0].text)
    return content["result"] if set(content) == {"result"} else content


@pytest.fixture
def mcp(embedded):
    return server.build(Index(embedded[0]))[0]


def test_lists_tools(mcp):
    async def go():
        async with Client(mcp) as client:
            return [t.name for t in (await client.list_tools()).tools]

    assert asyncio.run(go()) == ["search", "vsearch", "query", "get", "status"]


@pytest.mark.parametrize("tool", ["search", "vsearch", "query"])
def test_search_tools(mcp, tool):
    hits = data(call(mcp, tool, {"query": "brush strokes", "k": 1, "filters": {"categories": "cs.RO"}}))
    assert [h["id"] for h in hits] == ["notes/robots/brush.md"]
    assert set(hits[0]) == {"id", "title", "lines", "score", "text"}
    hits = data(call(mcp, tool, {"query": "reward", "collection": "notes", "path": "*.md", "k": 3}))
    assert "notes/rl.md" in [h["id"] for h in hits]


@pytest.mark.parametrize("tool", ["search", "vsearch", "query", "get", "status"])
def test_tools_run_on_the_event_loop_thread(embedded, monkeypatch, tool):
    # PyTorch crashes in worker threads on Python 3.15 (Windows, macOS), so no tool may run in one.
    index, threads = Index(embedded[0]), []
    real = getattr(index, tool)
    monkeypatch.setattr(index, tool, lambda *a, **kw: threads.append(threading.current_thread()) or real(*a, **kw))
    args = {"get": {"id": "notes/rl.md"}, "status": {}}.get(tool, {"query": "brush"})
    assert not call(server.build(index)[0], tool, args).is_error
    assert threads == [threading.main_thread()]


def test_get_and_status(mcp):
    result = call(mcp, "get", {"id": "notes/rl.md"})
    assert result.content[0].text.startswith("# Reward shaping")
    assert data(call(mcp, "status"))["collections"][0]["docs"] == 3


@pytest.mark.parametrize(
    "tool, args, reason",
    [
        ("search", {"query": "x", "collection": "missing"}, "no collection named 'missing'"),
        ("vsearch", {"query": "x", "k": 0}, "k must be a positive integer"),
        ("query", {"query": "x", "filters": {"bad key": "v"}}, "invalid filter field 'bad key'"),
        ("get", {"id": "missing/x.md"}, "unknown document 'missing/x.md'"),
    ],
)
def test_errors_tell_the_agent_what_went_wrong(mcp, tool, args, reason):
    # The MCP library hides the message of any exception that isn't a ToolError ("Error executing tool search"),
    # which leaves an agent no way to correct its call.
    result = call(mcp, tool, args)
    assert result.is_error
    assert reason in result.content[0].text


@pytest.mark.parametrize("stdio, expected", [(True, ("stdio", {})), (False, ("streamable-http", {"port": 9000}))])
def test_serve_loads_everything_then_runs(embedded, monkeypatch, stdio, expected):
    runs = []
    monkeypatch.setattr(MCPServer, "run", lambda self, transport, **kw: runs.append((transport, kw)))
    server.serve(port=9000, stdio=stdio)
    transport, kwargs = runs[0]
    assert transport == expected[0]
    assert expected[1].items() <= kwargs.items()


def test_serve_warns_without_embeddings(corpus, fake_models, monkeypatch, capsys):
    monkeypatch.setattr(MCPServer, "run", lambda self, transport, **kw: None)
    server.serve(stdio=True)
    assert "warning: no embeddings yet" in capsys.readouterr().err
