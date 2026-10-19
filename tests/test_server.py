import asyncio
import json
import logging
import sqlite3
import threading
import types

import anyio
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer

from sift import server
from sift.search import Index

from .conftest import write


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


def test_instructions_tell_agents_what_is_indexed(mcp):
    # Agents read these when they connect: without the collections they can't tell when sift is worth asking.
    assert "notes: 3 documents" in mcp.instructions
    assert "before answering from memory" in mcp.instructions


def test_instructions_name_the_front_matter_fields(mcp):
    # Agents can filter on these and ask for them in hits, if they know they exist.
    assert "- notes: 3 documents; front matter: categories, title, year" in mcp.instructions


def test_instructions_for_an_empty_index(tmp_path):
    from sift import store

    mcp = server.build(Index(store.connect(tmp_path / "empty")))[0]
    assert "No collections are indexed yet" in mcp.instructions


def test_server_log_stays_quiet(mcp):
    # The MCP library logs everything at INFO; model loading would then print dozens of HTTP requests.
    for name in ("httpx", "huggingface_hub", "sentence_transformers"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING, name


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


@pytest.fixture
def transports(monkeypatch):
    """Replaces the MCP transports with ones that serve while `serving()` is true, recording how they started."""
    fakes = types.SimpleNamespace(started=[], serving=lambda: False)

    def fake(name):
        async def run(self, **kwargs):
            fakes.started.append((name, kwargs))
            with anyio.fail_after(10):  # a safety net against hanging, not a timing assumption
                while fakes.serving():
                    await anyio.sleep(0)  # let the server's other tasks run

        return run

    monkeypatch.setattr(MCPServer, "run_stdio_async", fake("stdio"))
    monkeypatch.setattr(MCPServer, "run_streamable_http_async", fake("http"))
    return fakes


@pytest.mark.parametrize("stdio, expected", [(True, ("stdio", {})), (False, ("http", {"port": 9000}))])
def test_serve_loads_everything_then_runs(embedded, transports, stdio, expected):
    server.serve(port=9000, stdio=stdio)
    transport, kwargs = transports.started[0]
    assert transport == expected[0]
    assert expected[1].items() <= kwargs.items()


def test_serve_releases_the_gpu_when_idle(embedded, transports, monkeypatch):
    checks = []
    monkeypatch.setattr(server, "IDLE_CHECK_SECONDS", 0)
    monkeypatch.setattr(server.embed, "release_idle", lambda: checks.append(threading.current_thread()))
    transports.serving = lambda: len(checks) < 3  # serve until the idle check ran a few times
    server.serve(port=9000)
    assert len(checks) >= 3
    assert set(checks) == {threading.main_thread()}  # PyTorch must stay out of worker threads


def test_serve_warns_without_embeddings(corpus, fake_models, transports, capsys):
    server.serve(stdio=True)
    assert "warning: no embeddings yet" in capsys.readouterr().err


def test_database_errors_reach_the_agent(embedded, monkeypatch):
    index = Index(embedded[0])

    def locked(*args):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(index, "search", locked)
    result = call(server.build(index)[0], "search", {"query": "x"})
    assert result.is_error
    assert "database is locked" in result.content[0].text


@pytest.mark.parametrize("error", [KeyboardInterrupt, OSError])
def test_serve_raises_transport_errors_as_they_are(embedded, monkeypatch, error):
    # The idle check shares a task group with the transport, which wraps errors in an exception group:
    # Ctrl+C then printed a traceback instead of stopping quietly.
    async def failing(self, **kwargs):
        raise error("stop")

    monkeypatch.setattr(MCPServer, "run_stdio_async", failing)
    with pytest.raises(error, match="stop"):
        server.serve(stdio=True)


@pytest.fixture
def http_app(embedded, monkeypatch):
    """The HTTP app `sift serve` runs, captured instead of served."""
    import uvicorn

    apps = []

    async def capture(self, *args, **kwargs):
        apps.append(self.config.app)

    monkeypatch.setattr(uvicorn.Server, "serve", capture)
    server.serve(port=8765)
    return apps[0]


def test_http_server_accepts_only_local_hosts(http_app):
    # A web page can reach a local server under its own domain (DNS rebinding), so requests must name a local host
    # and come from a local page, if from any.
    from starlette.testclient import TestClient

    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    }
    with TestClient(http_app, base_url="http://127.0.0.1:8765") as client:
        assert client.post("/mcp", json=initialize, headers=headers).status_code == 200
        rebound = client.post("/mcp", json=initialize, headers=headers | {"Host": "attacker.example:8765"})
        assert rebound.status_code == 421
        foreign = client.post("/mcp", json=initialize, headers=headers | {"Origin": "http://attacker.example"})
        assert foreign.status_code == 403


@pytest.mark.parametrize("tool", ["search", "vsearch", "query"])
def test_search_tools_take_fields_compact_and_ranges(mcp, tool):
    args = {"query": "brush", "k": 1, "fields": ["year"], "compact": True, "filters": {"year": {">=": 2024}}}
    hits = data(call(mcp, tool, args))
    assert [(h["id"], h["fields"], "text" in h) for h in hits] == [("notes/robots/brush.md", {"year": 2024}, False)]


def test_get_takes_lines(mcp):
    result = call(mcp, "get", {"id": "notes/rl.md", "lines": "1"})
    assert result.content[0].text == "# Reward shaping\n"


def test_instructions_for_a_collection_without_front_matter(tmp_path):
    from sift import store

    folder = tmp_path / "plain"
    write(folder / "note.md", "# Note\n")
    db = store.connect(tmp_path / "home")
    store.add_collection(db, folder)
    store.index(db)
    assert "- plain: 1 document\n" in server.build(Index(db))[0].instructions + "\n"
