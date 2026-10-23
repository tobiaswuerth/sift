"""`sift serve`: the MCP tools agents call, the errors and instructions they read, and the server around them."""

import asyncio
import json
import logging
import shutil
import sqlite3
import threading
import types
from pathlib import Path

import anyio
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer

from sift import indexing, remote, server
from sift.search import Index

from .conftest import nobody_listens, write

SAMPLES = Path(__file__).parent / "samples"


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


def tools(mcp):
    async def go():
        async with Client(mcp) as client:
            return {t.name: t for t in (await client.list_tools()).tools}

    return asyncio.run(go())


def test_lists_tools(mcp):
    assert list(tools(mcp)) == [
        "search",
        "vsearch",
        "query",
        "similar",
        "get",
        "status",
        "links",
        "unresolved",
        "mentions",
        "reindex",
    ]


def test_every_parameter_is_described(mcp):
    # Agents see a tool's schema, not its code: a parameter without a description is one they guess at, or miss.
    for tool in tools(mcp).values():
        for name, schema in tool.input_schema["properties"].items():
            assert schema.get("description"), f"{tool.name}({name}) has no description"


def test_tools_are_marked_read_only(mcp):
    # Clients may then run them without asking the user each time. reindex changes only sift's index, never a
    # file, and running it twice does what running it once does.
    found = tools(mcp)
    assert all(tool.annotations.read_only_hint for name, tool in found.items() if name != "reindex")
    hints = found["reindex"].annotations
    assert (hints.read_only_hint, hints.destructive_hint, hints.idempotent_hint) == (False, False, True)


@pytest.mark.parametrize("tool", ["search", "vsearch", "query"])
def test_search_tools(mcp, tool):
    hits = data(call(mcp, tool, {"query": "brush strokes", "k": 1, "filters": {"categories": "cs.RO"}}))
    assert [h["id"] for h in hits] == ["notes/robots/brush.md"]
    assert set(hits[0]) == {"id", "title", "lines", "score", "text"}
    hits = data(call(mcp, tool, {"query": "reward", "collection": "notes", "path": "*.md", "k": 3}))
    assert "notes/rl.md" in [h["id"] for h in hits]


@pytest.mark.parametrize(
    "tool",
    ["search", "vsearch", "query", "similar", "get", "status", "links", "unresolved", "mentions", "reindex"],
)
def test_tools_run_on_the_event_loop_thread(embedded, monkeypatch, tool):
    # PyTorch crashes in worker threads on Python 3.15 (Windows, macOS), so no tool may run in one.
    index, threads = Index(embedded[0]), []
    real = getattr(index, tool)
    monkeypatch.setattr(index, tool, lambda *a, **kw: threads.append(threading.current_thread()) or real(*a, **kw))
    args = {name: {"id": "notes/rl.md"} for name in ("get", "similar", "links", "mentions")}
    args |= {"status": {}, "unresolved": {}, "reindex": {"paths": ["notes/rl.md"]}}
    args = args.get(tool, {"query": "brush"})
    assert not call(server.build(index)[0], tool, args).is_error
    assert threads == [threading.main_thread()]


def test_similar_tool(mcp):
    hits = data(call(mcp, "similar", {"id": "notes/brush.md", "k": 2, "fields": ["year"]}))
    assert len(hits) == 2
    assert "notes/robots/brush.md" not in [h["id"] for h in hits]


def test_instructions_tell_agents_what_is_indexed(mcp):
    # Agents read these when they connect: without the collections they can't tell when sift is worth asking.
    assert "): 3 documents" in mcp.instructions
    assert "before answering from memory" in mcp.instructions


def test_instructions_name_the_front_matter_fields(mcp, embedded):
    # Agents can filter on these and ask for them in hits, if they know they exist; and with the folder, they
    # know which collection a file they write belongs to.
    assert f"- notes ({embedded[1]}): 3 documents; front matter: categories, title, year" in mcp.instructions


def test_instructions_for_an_empty_index(tmp_path):
    from sift import store

    mcp = server.build(Index(store.connect(tmp_path / "empty")))[0]
    assert "No collections are indexed yet" in mcp.instructions


def test_server_log_stays_quiet(mcp):
    # The MCP library logs everything at INFO; model loading would then print dozens of HTTP requests.
    for name in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING, name


def test_get_and_status(mcp, embedded):
    result = call(mcp, "get", {"id": "notes/rl.md"})
    assert result.content[0].text.startswith("# Reward shaping")
    (collection,) = data(call(mcp, "status"))["collections"]
    assert (collection["docs"], collection["path"]) == (3, str(embedded[1]))


@pytest.mark.parametrize(
    "tool, args, reason",
    [
        ("search", {"query": "x", "collection": "missing"}, "no collection named 'missing'"),
        ("vsearch", {"query": "x", "k": 0}, "k must be a positive integer"),
        ("query", {"query": "x", "filters": {'bad"key': "v"}}, "invalid filter field 'bad\"key'"),
        ("get", {"id": "missing/x.md"}, "unknown document 'missing/x.md'"),
        ("links", {"id": "notes/rl.md", "direction": "up"}, "'out', 'in' or 'both'"),
        ("links", {"id": "notes/rl.md", "depth": 3}, "1 or 2"),
        ("unresolved", {"collection": "missing"}, "no collection named 'missing'"),
        ("reindex", {"paths": ["notes/.git/x.md"]}, "sift indexes no files in .git/"),
    ],
)
def test_errors_tell_the_agent_what_went_wrong(mcp, tool, args, reason):
    # The MCP library hides the message of any exception that isn't a ToolError ("Error executing tool search"),
    # which leaves an agent no way to correct its call.
    result = call(mcp, tool, args)
    assert result.is_error
    assert reason in result.content[0].text


HANS, ACME = "vault/01_People/Hans Müller.md", "vault/02_Organizations/Acme AG.md"


def test_links_tool(linked):
    mcp = server.build(Index(linked[0]))[0]
    found = data(call(mcp, "links", {"id": ACME, "direction": "both"}))
    assert [lk["id"] for lk in found["out"]] == [HANS]
    assert found["in"] == [
        {"id": HANS, "title": "Hans Müller", "links": [{"line": 3, "field": "org", "text": 'org: "[[Acme AG]]"'}]}
    ]
    found = data(call(mcp, "links", {"id": "vault/Anna Keller.md", "depth": 2, "field": "colleague_of", "k": 5}))
    assert [lk["id"] for lk in found["out"]] == [HANS]  # Hans links nothing through colleague_of


@pytest.mark.parametrize("tool", ["search", "query", "vsearch"])
def test_search_tools_take_links(linked, fake_models, tool):
    from sift import embed

    embed.embed_pending(linked[0])
    mcp = server.build(Index(linked[0]))[0]
    hits = data(call(mcp, tool, {"query": "Hans", "k": 1, "links_to": ["[[Acme AG]]"], "collection": "vault"}))
    assert [h["id"] for h in hits] == [HANS]
    hits = data(call(mcp, tool, {"query": "customer", "k": 1, "linked_from": ["vault/Hans Müller.md"]}))
    assert [h["id"] for h in hits] == [ACME]


def test_similar_tool_takes_links(linked, fake_models):
    from sift import embed

    embed.embed_pending(linked[0])
    mcp = server.build(Index(linked[0]))[0]
    offer = "vault/03_Interactions/2026-10-14 Offerte.md"
    hits = data(call(mcp, "similar", {"id": offer, "links_to": ["Acme AG"], "filters": {"org": "[[Acme AG]]"}}))
    assert [h["id"] for h in hits] == [HANS]


def test_reindex_tool(linked, fake_models):
    from sift import embed

    db, vault, _ = linked
    embed.embed_pending(db)
    mcp = server.build(Index(db))[0]
    write(vault / "06_Topics" / "Wartungsvertrag 2027.md", "Maintenance of the press line, from 2027.\n")
    found = data(call(mcp, "reindex", {"paths": [str(vault / "06_Topics" / "Wartungsvertrag 2027.md")]}))
    assert found == [{"id": "vault/06_Topics/Wartungsvertrag 2027.md", "status": "added", "chunks": 1, "embedded": 1}]
    hits = data(call(mcp, "vsearch", {"query": "maintenance of the press line", "k": 1}))
    assert [h["id"] for h in hits] == ["vault/06_Topics/Wartungsvertrag 2027.md"]


@pytest.mark.parametrize("ext", [".docx", ".pdf", ".html"])
def test_reindex_tool_reads_documents(linked, fake_models, ext):
    # Xberg reads documents in a coroutine of its own, and the tools run on the server's event loop.
    db, vault, _ = linked
    shutil.copy(SAMPLES / f"sample{ext}", vault / f"Offerte{ext}")
    mcp = server.build(Index(db))[0]
    (found,) = data(call(mcp, "reindex", {"paths": [str(vault / f"Offerte{ext}")]}))
    assert (found["id"], found["status"]) == (f"vault/Offerte{ext}", "added")
    assert f"This sample is the {ext[1:].upper()} file." in call(mcp, "get", {"id": found["id"]}).content[0].text


def test_mentions_tool(linked):
    mcp = server.build(Index(linked[0]))[0]
    hits = data(call(mcp, "mentions", {"id": HANS, "k": 3, "collection": "work", "also": ["aliases"]}))
    assert [h["id"] for h in hits] == ["work/contract.md"]
    hits = data(call(mcp, "mentions", {"id": HANS, "compact": True, "fields": ["*"]}))
    assert set(hits[0]) == {"id", "title", "lines", "score", "fields"}


def test_unresolved_tool(linked):
    mcp = server.build(Index(linked[0]))[0]
    assert [t["target"] for t in data(call(mcp, "unresolved", {"collection": "vault", "k": 1}))] == ["Peter Frei"]


def test_instructions_count_links(linked):
    # Agents should know which collections hold linked notes, and how many of their links lead nowhere yet.
    mcp = server.build(Index(linked[0]))[0]
    vault, work = linked[1:]
    lines = mcp.instructions.split("\n")
    assert (
        f"- vault ({vault}): 4 documents; front matter: aliases, colleague_of, org, reports_to; 6 links, 2 unresolved"
        in lines
    )
    assert f"- work ({work}): 1 document" in lines
    assert "`links`" in mcp.instructions


@pytest.fixture
def transports(monkeypatch):
    """Replaces the stdio transport and the HTTP server (uvicorn) with fakes that serve while `serving()` is true,
    recording how they were started."""
    import uvicorn

    fakes = types.SimpleNamespace(started=[], serving=lambda: False)

    async def serving():
        with anyio.fail_after(10):  # a safety net against hanging, not a timing assumption
            while fakes.serving():
                await anyio.sleep(0)  # let the server's other tasks run

    async def stdio(self):
        fakes.started.append(("stdio", {}))
        await serving()

    async def http(self, sockets=None):
        fakes.started.append(("http", {"host": self.config.host, "port": self.config.port}))
        await serving()

    monkeypatch.setattr(MCPServer, "run_stdio_async", stdio)
    monkeypatch.setattr(uvicorn.Server, "serve", http)
    return fakes


@pytest.mark.parametrize("host, warned", [("0.0.0.0", True), ("192.168.1.5", True), ("127.0.0.1", False)])
def test_serving_other_machines_warns_that_they_can_read_everything(embedded, transports, capsys, host, warned):
    server.serve(host=host, port=9000)
    assert ("anyone who reaches port 9000 can read every indexed document" in capsys.readouterr().err) == warned


@pytest.mark.parametrize(
    "stdio, expected", [(True, ("stdio", {})), (False, ("http", {"host": "127.0.0.1", "port": 9000}))]
)
def test_serve_loads_everything_then_runs(embedded, transports, stdio, expected):
    server.serve(port=9000, stdio=stdio)
    transport, kwargs = transports.started[0]
    assert transport == expected[0]
    assert expected[1].items() <= kwargs.items()


def test_serve_releases_the_gpu_when_idle(embedded, fake_models, transports, monkeypatch):
    checks = []
    monkeypatch.setattr(server, "IDLE_CHECK_SECONDS", 0)
    monkeypatch.setattr(fake_models, "release_idle", lambda: checks.append(threading.current_thread()))
    transports.serving = lambda: len(checks) < 3  # serve until the idle check ran a few times
    server.serve(port=9000)
    assert len(checks) >= 3
    assert set(checks) == {threading.main_thread()}  # PyTorch must stay out of worker threads


def test_serve_warns_without_embeddings(corpus, fake_models, transports, capsys):
    server.serve(stdio=True)
    assert "warning: no embeddings yet" in capsys.readouterr().err


def test_serve_with_a_remote_loads_no_models(embedded, transports, fake_models, monkeypatch, configure):
    # The point of a remote: this machine may be too weak to run the models at all.
    configure(remote="https://t.p@workstation:8766")
    monkeypatch.setattr(remote.Remote, "load", lambda self: None)
    monkeypatch.setattr(fake_models, "embedder", lambda: pytest.fail("loaded the embedding model here"))
    monkeypatch.setattr(fake_models, "reranker", lambda: pytest.fail("loaded the reranker here"))
    server.serve(stdio=True)
    assert transports.started


def test_serve_warns_when_the_remote_is_unreachable(embedded, transports, fake_models, capsys, configure):
    configure(remote=nobody_listens())
    server.serve(stdio=True)  # keyword search works without it
    assert "warning: embedding remote" in capsys.readouterr().err


def test_an_unreachable_remote_reaches_the_agent(mcp, fake_models, configure):
    configure(remote=nobody_listens())
    result = call(mcp, "vsearch", {"query": "brush"})
    assert result.is_error
    assert "embedding remote" in result.content[0].text


def test_database_errors_reach_the_agent(embedded, monkeypatch):
    index = Index(embedded[0])

    def locked(*args, **kwargs):
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


@pytest.mark.parametrize(
    "tool, first", [("search", "query"), ("vsearch", "query"), ("query", "query"), ("similar", "id")]
)
def test_search_tools_pass_every_option_on(embedded, monkeypatch, tool, first):
    index, calls = Index(embedded[0]), []
    monkeypatch.setattr(index, tool, lambda *args, **kwargs: calls.append((args, kwargs)) or [])
    options = {
        "collection": "notes",
        "path": "*.md",
        "filters": {"year": 2024},
        "fields": ["year"],
        "compact": True,
        "all_passages": True,
        "sort": "-year",
        "links_to": ["a"],
        "linked_from": ["b"],
    }
    call(server.build(index)[0], tool, {first: "brush", "k": 3} | options)
    assert calls == [(("brush", 3), options)]


@pytest.mark.parametrize("lines", ["1", "1-1", 1])
def test_get_takes_lines(mcp, lines):
    result = call(mcp, "get", {"id": "notes/rl.md", "lines": lines})
    assert result.content[0].text == "# Reward shaping\n"


def test_get_with_fields_returns_the_document_with_them(mcp):
    document = data(call(mcp, "get", {"id": "notes/brush.md", "lines": "1-3", "fields": ["year"]}))
    assert document == {
        "id": "notes/robots/brush.md",
        "title": "Brush stroke planning",
        "fields": {"year": 2024},
        "text": '---\ntitle: "Brush stroke planning"\ncategories: ["cs.RO", "cs.GR"]\n',
    }


def test_unexpected_errors_reach_the_agent_too(embedded, monkeypatch, caplog):
    # A bug is still an error the agent should see: "Error executing tool search" alone says nothing.
    index = Index(embedded[0])
    monkeypatch.setattr(index, "search", lambda *args, **kwargs: 1 / 0)
    result = call(server.build(index)[0], "search", {"query": "x"})
    assert result.is_error
    assert "ZeroDivisionError: division by zero" in result.content[0].text
    assert "Traceback" in caplog.text  # for whoever fixes it


def test_instructions_for_a_collection_without_front_matter(tmp_path):
    from sift import store

    folder = tmp_path / "plain"
    write(folder / "note.md", "# Note\n")
    db = store.connect(tmp_path / "home")
    store.add_collection(db, folder)
    indexing.index(db)
    assert f"- plain ({folder.resolve()}): 1 document\n" in server.build(Index(db))[0].instructions + "\n"


def test_http_shutdown_gives_up_on_unfinished_requests(embedded, fake_models, monkeypatch, capsys):
    # On shutdown, uvicorn waits for every connection to close (Python 3.12+), and a second Ctrl+C doesn't change
    # that: a client that left a request unfinished kept `sift serve` from ever stopping.
    import uvicorn

    monkeypatch.setattr(server, "SHUTDOWN_SECONDS", 0.5)
    servers, serve = [], uvicorn.Server.serve

    async def recording(self, sockets=None):
        servers.append(self)
        await serve(self, sockets)

    monkeypatch.setattr(uvicorn.Server, "serve", recording)
    mcp, index = server.build(Index(embedded[0]))

    async def main():
        with anyio.fail_after(10):  # a hang detector: the server stops within half a second
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(server._serve, mcp, index, "127.0.0.1", 0, False, fake_models)
                while not (servers and servers[0].started):
                    await anyio.sleep(0.01)
                port = servers[0].servers[0].sockets[0].getsockname()[1]
                client = await anyio.connect_tcp("127.0.0.1", port)
                await client.send(b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 100\r\n\r\n{")
                while not servers[0].server_state.tasks:  # the request is in, its body never will be
                    await anyio.sleep(0.01)
                servers[0].should_exit = True  # what Ctrl+C does
            await client.aclose()

    anyio.run(main)
    err = capsys.readouterr().err
    assert "Cancel 1 running task(s)" in err
    assert "Traceback" not in err  # sift cancelled the request on purpose; the line above says so
