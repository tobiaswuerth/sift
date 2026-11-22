"""`sift serve`: the MCP tools agents call, the errors and instructions they read, and the server around them."""

import asyncio
import dataclasses
import json
import logging
import re
import shutil
import socket
import sqlite3
import sys
import threading
import types
from pathlib import Path

import anyio
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer

from sift import indexing, logs, remote, server, vectors, worker
from sift.catalog import Catalog
from sift.hits import Options
from sift.search import Index

from .conftest import nobody_listens, on_windows, write

port_free = server.port_free  # sift's own, before a fixture keeps the tests off this machine's ports (ports_unasked)

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


def test_every_search_tool_takes_every_option_of_a_search(mcp):
    # A search's options were written out for each search and each tool, nine times: one added to the searches
    # reached no agent until the tools were given it too, by hand.
    options = {option.name for option in dataclasses.fields(Options)}
    for name in ("search", "vsearch", "query", "similar"):
        assert options <= set(tools(mcp)[name].input_schema["properties"]), name
    assert set(server.OPTIONS) == options | {"budget"}


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
def test_tools_run_off_the_event_loop_and_the_models_on_it(embedded, monkeypatch, tool):
    # A tool's work runs on a worker thread, so the event loop stays free to answer and to hear cancellations; the
    # models still run on the main thread, which holds the loop: PyTorch crashes in worker threads on Python 3.15
    # (Windows, macOS).
    index, threads, model_threads = Index(embedded[0]), [], []
    real = getattr(index, tool)
    monkeypatch.setattr(index, tool, lambda *a, **kw: threads.append(threading.current_thread()) or real(*a, **kw))
    mcp = server.build(index)[0]
    for name in ("encode", "rerank"):
        method = getattr(embedded_models(index), name)
        monkeypatch.setattr(
            embedded_models(index),
            name,
            lambda *a, _m=method, **kw: model_threads.append(threading.current_thread()) or _m(*a, **kw),
        )
    args = {name: {"id": "notes/rl.md"} for name in ("get", "similar", "links", "mentions")}
    args |= {"status": {}, "unresolved": {}, "reindex": {"paths": ["notes/rl.md"]}}
    args = args.get(tool, {"query": "brush"})
    assert not call(mcp, tool, args).is_error
    assert len(threads) == 1
    assert threads[0] is not threading.main_thread()
    assert set(model_threads) <= {threading.main_thread()}
    if tool in ("vsearch", "query"):
        assert model_threads


def embedded_models(index):
    """The models behind the server's wrapper, which runs them on the main thread."""
    return index.models._models


# Endless for a test: tens of seconds. Truly endless, a regression would hang the test run rather than fail it.
ENDLESS = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n LIMIT 300000000) SELECT count(*) FROM n"


def slow_tool(index, monkeypatch, began, stopped):
    """Make `search` run a statement that never ends by itself, on a store's connection."""
    db = index.catalog.store(next(iter(index.active().values())))

    def endless(*args, **kwargs):
        began.set()
        try:
            db.execute(ENDLESS)
        except sqlite3.OperationalError as e:
            stopped.append(e)
            raise

    monkeypatch.setattr(index, "search", endless)


def test_the_server_answers_while_a_tool_runs(embedded, monkeypatch):
    # A slow search kept the event loop busy: the server answered no one, not even a client connecting anew.
    index, began, stopped = Index(embedded[0]), threading.Event(), []
    slow_tool(index, monkeypatch, began, stopped)
    mcp = server.build(index)[0]

    async def main():
        async with Client(mcp) as client, anyio.create_task_group() as tasks:
            tasks.start_soon(client.call_tool, "search", {"query": "x"})
            with anyio.fail_after(5):
                while not began.is_set():
                    await anyio.sleep(0.01)
                listed = await client.list_tools()  # answered while the search runs
            tasks.cancel_scope.cancel()
        return listed

    assert len(anyio.run(main).tools) == 10


def test_a_cancelled_search_stops_and_the_next_call_runs(embedded, monkeypatch):
    # An agent that gave up on a search (its timeout) left it running for minutes, and every later call waited.
    index, began, stopped = Index(embedded[0]), threading.Event(), []
    slow_tool(index, monkeypatch, began, stopped)
    mcp = server.build(index)[0]

    async def main():
        async with Client(mcp) as client:
            with anyio.move_on_after(5) as gave_up:
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(client.call_tool, "search", {"query": "x"})
                    while not began.is_set():
                        await anyio.sleep(0.01)
                    tasks.cancel_scope.cancel()  # what a client's timeout does
            assert not gave_up.cancel_called
            with anyio.fail_after(5):
                return await client.call_tool("status", {})

    assert not anyio.run(main).is_error
    assert "interrupted" in str(stopped[0])


def test_similar_tool(mcp):
    hits = data(call(mcp, "similar", {"id": "notes/brush.md", "k": 2, "fields": ["year"]}))
    assert len(hits) == 2
    assert "notes/robots/brush.md" not in [h["id"] for h in hits]


def test_instructions_tell_agents_what_is_indexed(mcp):
    # Agents read these when they connect: without the collections they can't tell when sift is worth asking.
    assert "): 3 documents" in mcp.instructions
    assert "before answering from memory" in mcp.instructions


def test_instructions_say_which_tool_fits_what_the_agent_has(mcp):
    # Agents searched exact strings, a version or an error code, by meaning, and questions by keyword.
    guide = mcp.instructions
    assert "A question, topic or description: `query`" in guide
    assert 'a version "3.12.4", an error code' in guide
    assert "wording you can't guess: `vsearch`" in guide


def test_instructions_teach_research_in_rounds(mcp):
    # One search, its first result read, and an answer: agents missed what other phrasings and the hits' own terms
    # find. The arguments are written as calls take them, so that weaker models copy them.
    guide = mcp.instructions
    assert 'query=["...", "...", "..."]' in guide  # several phrasings, one call
    assert "`compact: true, k: 30, adaptive: true`" in guide  # many results surveyed, few read
    assert 'id=[{"id": "...", "lines": "17-26"}, ...]' in guide  # the best few read, one call
    assert "`similar` and `links` on the best one" in guide
    assert "Stop when a round returns only documents you have seen" in guide


def test_instructions_put_how_to_search_before_the_collections(mcp):
    # Claude Code is reported to keep the first 2,048 characters of a server's instructions. The collections, as
    # many as are indexed, come last, so a long list cuts off only its own end, and `status` lists them all.
    guide, collections = mcp.instructions.split("Indexed collections")
    assert len(guide) <= 1500
    assert collections.startswith(" (`status` lists them all, with their fields):\n- notes (")


def test_tool_descriptions_fit_what_clients_keep(mcp):
    # Claude Code is reported to cut each tool's description at 2,048 characters, saying nothing.
    for tool in tools(mcp).values():
        assert len(tool.description) <= 2048, tool.name


def test_every_tool_returns_its_result_as_data_too(mcp):
    # status and links returned text alone, the others data beside it (structured content): a client that hands its
    # model the data, compact, handed it their text, indented, twice the size and more. A return type alone decides
    # it, so each tool's is held here.
    assert all(tool.output_schema for tool in tools(mcp).values())
    for tool, args in (("status", {}), ("links", {"id": "notes/rl.md"})):
        assert isinstance(call(mcp, tool, args).structured_content, dict), tool


@pytest.mark.parametrize("tool", ["search", "vsearch", "query", "similar"])
def test_search_tools_say_how_to_read_on(mcp, tool):
    # An agent that found a tool by tool search reads its description alone, not the server's instructions.
    description = tools(mcp)[tool].description
    assert "`get` the best few by id and lines, in one call" in description
    assert "`also`" in description


def test_keyword_and_hybrid_search_point_to_each_other(mcp):
    # Each says when the other fits: exact strings, or questions.
    found = tools(mcp)
    assert "For a question or topic, use `query`" in found["search"].description
    assert 'For an exact string, e.g. a version "3.12.4", an error code or a name, use `search`' in (
        found["query"].description
    )
    assert "pass several phrasings at once, as a list" in found["query"].description


def test_instructions_name_the_front_matter_fields(mcp, embedded):
    # Agents can filter on these and ask for them in hits, if they know they exist; and with the folder, they
    # know which collection a file they write belongs to.
    assert f"- notes ({embedded[1]}): 3 documents; front matter: categories, title, year" in mcp.instructions


def test_instructions_for_an_empty_index(tmp_path):

    mcp = server.build(Index(Catalog(tmp_path / "empty")))[0]
    assert "No collections are indexed yet" in mcp.instructions


def test_server_log_stays_quiet(mcp):
    # The MCP library logs everything at INFO; model loading would then print dozens of HTTP requests.
    for name in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING, name


def test_the_log_keeps_that_a_tool_failed_not_what_it_was_asked(mcp, home):
    # The MCP library logs why each tool failed, as sift told the caller: "unknown document '...'" names the document
    # asked for, kept on disk in SIFT_HOME/logs for weeks, where the requests' queries are left out.
    from datetime import datetime

    with logs.kept("serve", datetime(2026, 11, 22, 12, 0, 0)) as path:
        assert call(mcp, "get", {"id": "notes/wombatsecret.md"}).is_error
        assert call(mcp, "mentions", {"id": "wombatsecret"}).is_error
    written = path.read_text(encoding="utf-8")
    assert "Tool 'get' failed" in written
    assert "Tool 'mentions' failed" in written
    assert "wombatsecret" not in written
    other = logging.makeLogRecord({"msg": "Tool %r rejected arguments: %r", "args": ("get", ["id"])})
    assert server._no_reasons(other)
    assert other.getMessage() == "Tool 'get' rejected arguments: ['id']"  # which names no value


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
        ("query", {"query": "x", "filters": {"": "v"}}, "invalid filter field ''"),
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

    catalog, vault, _ = linked
    embed.embed_pending(catalog)
    mcp = server.build(Index(catalog))[0]
    write(vault / "06_Topics" / "Wartungsvertrag 2027.md", "Maintenance of the press line, from 2027.\n")
    found = data(call(mcp, "reindex", {"paths": [str(vault / "06_Topics" / "Wartungsvertrag 2027.md")]}))
    assert found == [{"id": "vault/06_Topics/Wartungsvertrag 2027.md", "status": "added", "chunks": 1, "embedded": 1}]
    hits = data(call(mcp, "vsearch", {"query": "maintenance of the press line", "k": 1}))
    assert [h["id"] for h in hits] == ["vault/06_Topics/Wartungsvertrag 2027.md"]


@pytest.mark.parametrize("ext", [".docx", ".pdf", ".html"])
def test_reindex_tool_reads_documents(linked, fake_models, ext):
    # Xberg reads documents in a coroutine of its own, and the tools run on the server's event loop.
    catalog, vault, _ = linked
    shutil.copy(SAMPLES / f"sample{ext}", vault / f"Offerte{ext}")
    mcp = server.build(Index(catalog))[0]
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


def test_a_port_that_is_taken_is_said_to_be():
    # `sift serve` beside one that runs already loaded the models first, for minutes on a CPU, and then ended with
    # uvicorn's "[Errno 10048] error while attempting to bind on address".
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        said = f"port {port} is taken: another `sift serve` runs there already, which agents can use as it is, or "
        with pytest.raises(OSError, match=re.escape(said + "another program; `--port` picks another")):
            port_free("127.0.0.1", port)
    port_free("127.0.0.1", port)  # free again
    nowhere = "nothing can listen on 192.0.2.1, port 8765: it is no address of this machine"  # nor of any other
    with pytest.raises(OSError, match=re.escape(nowhere)):
        port_free("192.0.2.1", 8765)
    with pytest.raises(OSError, match="there is no port 70000: `--port` takes a number up to 65535"):
        port_free("127.0.0.1", 70000)


def test_a_port_is_asked_for_at_each_address_of_its_host_as_a_server_takes_them(monkeypatch):
    # asyncio passes over an address whose family the machine has not, as IPv6 where it is off, and listens on the
    # others: "localhost" must not be refused for its ::1.
    local = (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))
    no_family = (9999, socket.SOCK_STREAM, 0, "", ("::1", 0, 0, 0))  # no family any machine has
    not_ours = (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.0.2.1", 0))
    monkeypatch.setattr(server.socket, "getaddrinfo", lambda *args, **kwargs: [no_family, not_ours, local])
    port_free("localhost", 8765)
    monkeypatch.setattr(server.socket, "getaddrinfo", lambda *args, **kwargs: [no_family])
    with pytest.raises(OSError, match="nothing can listen on localhost, port 8765: it is no address of this machine"):
        port_free("localhost", 8765)

    def unknown(*args, **kwargs):
        raise socket.gaierror(11001, "getaddrinfo failed")

    monkeypatch.setattr(server.socket, "getaddrinfo", unknown)
    with pytest.raises(OSError, match=r"nothing can listen on nas, port 8765: .*getaddrinfo failed"):
        port_free("nas", 8765)


@pytest.mark.parametrize("platform, reuses", [("win32", False), ("linux", True)])
def test_a_port_is_asked_for_as_asyncio_takes_one_for_a_server(monkeypatch, platform, reuses):
    # A port a server just left is free again everywhere but on Windows (SO_REUSEADDR), where the same option shares
    # a port in use: asked for otherwise, a port taken passed there, or one just left was refused elsewhere.
    options = []

    class Probe(socket.socket):
        def setsockopt(self, *option):
            options.append(option)
            super().setsockopt(*option)

        def bind(self, address):
            if address[1] == 80:
                raise PermissionError(13, "Permission denied")  # as Linux keeps the low ports from a user
            super().bind(address)

    with monkeypatch.context() as elsewhere:  # for sift alone: pytest goes on, on the system it runs on
        elsewhere.setattr(server.socket, "socket", Probe)
        elsewhere.setattr(sys, "platform", platform)
        port_free("127.0.0.1", 0)
        with pytest.raises(OSError, match=r"nothing can listen on 127\.0\.0\.1, port 80: .*Permission denied"):
            port_free("127.0.0.1", 80)
    assert options == [(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)] * (2 * reuses)


@pytest.mark.parametrize("serve", [server.serve, worker.serve])
def test_a_server_asks_for_its_port_before_it_loads_the_models(home, transports, fake_models, monkeypatch, serve):
    def taken(host, port):
        raise OSError(f"port {port} is taken")

    monkeypatch.setattr(serve.__globals__["__name__"] + ".port_free", taken)
    monkeypatch.setattr(fake_models, "load", lambda: pytest.fail("the models were loaded"))
    with pytest.raises(OSError, match="port 9000 is taken"):
        serve(port=9000, models=fake_models)
    assert transports.started == []


def test_a_server_over_stdio_asks_for_no_port(embedded, transports, monkeypatch):
    monkeypatch.setattr(server, "port_free", lambda host, port: pytest.fail("asked for a port"))
    server.serve(stdio=True)
    assert transports.started == [("stdio", {})]


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


def test_serve_loads_the_compiled_scan_up_front(embedded, transports, monkeypatch):
    # Loading Numba costs a second: a server pays it as it starts, a single search from the command line not at all.
    monkeypatch.setattr(vectors, "_scan", None)
    server.serve(stdio=True)
    assert vectors._scan is not None


def test_serve_without_the_compiled_scan_says_so_and_scans_with_numpy(embedded, transports, monkeypatch, capsys):
    from sift import scan

    def broken(*args):
        raise RuntimeError("cannot cache function '_scan': no locator available")

    monkeypatch.setattr(vectors, "_scan", None)
    monkeypatch.setattr(scan, "measure", broken)
    server.serve(stdio=True)
    assert vectors._scan is None
    assert "the compiled scan of the codes isn't available (cannot cache" in capsys.readouterr().err
    assert transports.started


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


def test_serve_with_a_remote_loads_no_models(embedded, transports, fake_models, monkeypatch, configure, capsys):
    # The point of a remote: this machine may be too weak to run the models at all.
    configure(remote="https://t.p@workstation:8766")
    monkeypatch.setattr(remote.Remote, "load", lambda self: None)
    monkeypatch.setattr(fake_models, "embedder", lambda: pytest.fail("loaded the embedding model here"))
    monkeypatch.setattr(fake_models, "reranker", lambda: pytest.fail("loaded the reranker here"))
    server.serve(stdio=True)
    assert transports.started
    said = capsys.readouterr().err  # it said it loaded the models into RAM, where it reaches the worker
    assert "getting the models on https://workstation:8766 ready and loading the vectors' codes" in said


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


def test_on_windows_http_is_served_on_a_selector_loop(embedded, monkeypatch):
    # Windows' default loop, the proactor, closes the listening socket for good when one accept fails (WinError 64,
    # a client gone mid-accept): with 20 agents connecting at once, sift ran on but answered no one.
    import uvicorn

    loops = []

    async def capture(self, *args, **kwargs):
        loops.append(asyncio.get_running_loop())

    monkeypatch.setattr(uvicorn.Server, "serve", capture)
    monkeypatch.setattr(server, "loop_options", on_windows(server.loop_options))
    server.serve(port=8765)
    assert isinstance(loops[0], asyncio.SelectorEventLoop)


def test_the_loop_is_chosen_for_windows_and_http_only():
    assert server.loop_options(platform="win32") == {"loop_factory": asyncio.SelectorEventLoop}
    assert server.loop_options(stdio=True, platform="win32") == {}  # stdio's pipes are the proactor's to read
    assert server.loop_options(platform="linux") == {}  # elsewhere the default loop is a selector loop already


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


def test_a_store_that_refuses_to_open_is_a_tool_error(mcp, monkeypatch):
    # A newer sift's store, opened by a server still running: SystemExit went through the tool, and the server.
    real = Catalog.store

    def refused(self, collection):
        raise SystemExit("notes.sqlite was made by a newer sift; upgrade sift to use it")

    monkeypatch.setattr(Catalog, "store", refused)
    result = call(mcp, "search", {"query": "brush"})
    assert result.is_error
    assert "made by a newer sift" in result.content[0].text
    monkeypatch.setattr(Catalog, "store", real)
    assert data(call(mcp, "search", {"query": "brush"}))  # the next call answers


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
        "adaptive": True,
    } | ({"budget": 0.5} if tool == "search" else {})  # only keyword search has a time budget
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


def test_get_takes_several_documents_at_once(mcp):
    # An agent reading the hits of a search spent a call, and a turn of its own, on each.
    documents = data(
        call(
            mcp,
            "get",
            {"id": [{"id": "notes/rl.md", "lines": "1"}, "notes/brush.md", "missing/x.md"], "fields": ["year"]},
        )
    )
    assert documents[0] == {"id": "notes/rl.md", "title": "Reward shaping", "fields": {}, "text": "# Reward shaping\n"}
    assert (documents[1]["id"], documents[1]["fields"]) == ("notes/robots/brush.md", {"year": 2024})
    assert documents[1]["text"].startswith("---\ntitle:")
    assert documents[2] == {"id": "missing/x.md", "error": "unknown document 'missing/x.md'"}  # the others still come


def test_a_batch_leaves_out_texts_past_its_budget(mcp, monkeypatch):
    monkeypatch.setattr(server, "BATCH_TEXT", 30)
    asked = [{"id": "notes/rl.md", "lines": 1}, "notes/rl.md", {"id": "notes/brush.md", "lines": "1"}]
    documents = data(call(mcp, "get", {"id": asked}))
    assert [d.get("text") for d in documents] == ["# Reward shaping\n", None, "---\n"]  # 17 and 4 of 30 characters
    assert "30 characters" in documents[1]["omitted"]
    assert "lines" in documents[1]["omitted"]
    assert set(documents[1]) == {"id", "title", "omitted"}


@pytest.mark.parametrize(
    "ids, lines, reason",
    [([], None, "at least one"), (["notes/rl.md"] * 101, None, "at most 100"), (["notes/rl.md"] * 2, "x", "lines")],
)
def test_a_batch_names_what_it_takes(mcp, ids, lines, reason):
    # Lines shared by every document, and wrong, were one error each instead of one for the call.
    result = call(mcp, "get", {"id": ids, "lines": lines})
    assert result.is_error
    assert reason in result.content[0].text


@pytest.mark.parametrize("tool", ["search", "vsearch", "query"])
def test_search_tools_take_several_queries(embedded, tool):
    asked = ["brush strokes", "reward shaping"]
    both = data(call(server.build(Index(embedded[0]))[0], tool, {"query": asked, "k": 1, "compact": True}))
    alone = [
        data(call(server.build(Index(embedded[0]))[0], tool, {"query": q, "k": 1, "compact": True})) for q in asked
    ]
    assert both == alone  # each asked of a server of its own: nothing kept from the others


def test_a_document_of_several_takes_only_id_and_lines(mcp):
    result = call(mcp, "get", {"id": [{"id": "notes/rl.md", "line": "1"}]})  # "line" for "lines"
    assert result.is_error
    assert "line" in result.content[0].text


@pytest.mark.parametrize("bug", [KeyError("fields"), IndexError("list index out of range")])
def test_a_bug_that_is_a_lookup_error_is_logged_as_a_bug(embedded, monkeypatch, caplog, bug):
    # Both are LookupErrors, as "unknown document ..." is: the agent got "'fields'", and the log nothing.
    def broken(*args, **kwargs):
        raise bug

    index = Index(embedded[0])
    monkeypatch.setattr(index, "search", broken)
    result = call(server.build(index)[0], "search", {"query": "x"})
    assert result.is_error
    assert f"{type(bug).__name__}: " in result.content[0].text  # after the library's "Error executing tool search: "
    assert "Traceback" in caplog.text


def test_unexpected_errors_reach_the_agent_too(embedded, monkeypatch, caplog):
    # A bug is still an error the agent should see: "Error executing tool search" alone says nothing.
    index = Index(embedded[0])
    monkeypatch.setattr(index, "search", lambda *args, **kwargs: 1 / 0)
    result = call(server.build(index)[0], "search", {"query": "x"})
    assert result.is_error
    assert "ZeroDivisionError: division by zero" in result.content[0].text
    assert "Traceback" in caplog.text  # for whoever fixes it


def test_instructions_for_a_collection_without_front_matter(tmp_path):

    folder = tmp_path / "plain"
    write(folder / "note.md", "# Note\n")
    catalog = Catalog(tmp_path / "home")
    catalog.add(folder)
    indexing.index(catalog)
    assert f"- plain ({folder.resolve()}): 1 document\n" in server.build(Index(catalog))[0].instructions + "\n"


def test_over_http_a_call_the_client_abandons_stops(embedded, fake_models, monkeypatch):
    # How agents connect: over HTTP, a client that gives up on a call closes its request, which must stop the call.
    import uvicorn

    servers, serve = [], uvicorn.Server.serve

    async def recording(self, sockets=None):
        servers.append(self)
        await serve(self, sockets)

    monkeypatch.setattr(uvicorn.Server, "serve", recording)
    index, began, stopped = Index(embedded[0]), threading.Event(), []
    slow_tool(index, monkeypatch, began, stopped)
    mcp, index = server.build(index)

    async def main():
        with anyio.fail_after(20):  # a hang detector
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(server._serve, mcp, index, "127.0.0.1", 0, False, fake_models)
                while not (servers and servers[0].started):
                    await anyio.sleep(0.01)
                url = f"http://127.0.0.1:{servers[0].servers[0].sockets[0].getsockname()[1]}/mcp"
                async with Client(url) as client:
                    async with anyio.create_task_group() as calls:
                        calls.start_soon(client.call_tool, "search", {"query": "x"})
                        while not began.is_set():
                            await anyio.sleep(0.01)
                        calls.cancel_scope.cancel()  # the client's timeout
                    status = await client.call_tool("status", {})
                servers[0].should_exit = True
        return status

    assert not anyio.run(main).is_error
    assert "interrupted" in str(stopped[0])


def test_the_log_of_sift_serve_keeps_no_query(embedded, fake_models, monkeypatch, home):
    # Every line of uvicorn's access log held the request's query: each UI search ("?q=..."), each document read
    # ("?doc=...", "?id=..."), kept on disk in SIFT_HOME/logs for weeks. The path and the answer stay.
    import urllib.request
    from datetime import datetime

    import uvicorn

    servers, serve = [], uvicorn.Server.serve

    async def recording(self, sockets=None):
        servers.append(self)
        await serve(self, sockets)

    monkeypatch.setattr(uvicorn.Server, "serve", recording)
    mcp, index = server.build(Index(embedded[0]))

    async def main():
        with anyio.fail_after(20):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(server._serve, mcp, index, "127.0.0.1", 0, False, fake_models)
                while not (servers and servers[0].started):
                    await anyio.sleep(0.01)
                port = servers[0].servers[0].sockets[0].getsockname()[1]
                url = f"http://127.0.0.1:{port}/ui/?q=wombatsecret&doc=notes%2Frl.md"
                await anyio.to_thread.run_sync(lambda: urllib.request.urlopen(url, timeout=10).close())
                servers[0].should_exit = True

    with logs.kept("serve", datetime(2026, 11, 21, 12, 0, 0)) as path:
        anyio.run(main)
    written = path.read_text(encoding="utf-8")
    assert '"GET /ui/ HTTP/1.1" 200' in written
    assert "wombatsecret" not in written
    assert "rl.md" not in written
    other = logging.makeLogRecord({"msg": "%s", "args": ("a line of another shape?x=1",)})
    assert server._no_queries(other)
    assert other.args == ("a line of another shape?x=1",)  # left as it is


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
