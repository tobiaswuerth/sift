import dataclasses
import http.client
import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.errors import InternalError, SystemFailureError
from sift.api.serving import WorkerAddress
from sift.api.wire import Json
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.indexing import NO_COLLECTION_REMEDY
from sift.interfaces.acts.acting import Acting
from sift.interfaces.mcp.instructions import HOW
from sift.interfaces.mcp.tools import Tool, ToolRunning
from sift.interfaces.server.agents import AgentsRunning, AgentsServed
from sift.interfaces.server.app import FAILED_REMEDY
from sift.interfaces.server.handled import HANDLED, Handled
from sift.interfaces.server.serving import HomeServer
from sift.settings import Settings
from sift.storage.store_files import CATALOG
from tests.damage import damage_pages
from tests.files import write
from tests.homes import indexed, quiet
from tests.models import Hashed
from tests.serving import asked, last, running
from tests.workers import LOOPBACK, nobody_listens, working

type Message = dict[str, object]


def posted(
    server: HomeServer, body: object, headers: Mapping[str, str] | None = None, method: str = "POST"
) -> tuple[int, Mapping[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", server.endpoint.port, timeout=60)
    sent = body if isinstance(body, bytes) else json.dumps(body).encode()
    given = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", **(headers or {})}
    try:
        connection.request(method, "/mcp", sent, given)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def answer(server: HomeServer, method: str, params: object) -> Message:
    status, _, read = posted(server, {"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    assert status == 200, read
    return cast("Message", json.loads(read))


def result(server: HomeServer, method: str, params: object) -> Message:
    return cast("Message", answer(server, method, params)["result"])


@pytest.fixture
def home(settings: Settings, notes: Path, tmp_path: Path) -> Settings:
    add(settings.home, AddCollection(str(notes)))
    write(tmp_path / "hidden" / "secret.md", "# Secret\n\nDark matter of a secret kind.\n")
    add(settings.home, AddCollection(str(tmp_path / "hidden"), agents=False))
    indexed(settings.home)
    return settings


def test_an_agent_connects_without_a_token_and_is_told_what_it_may_search(home: Settings) -> None:
    with running(home) as server:
        hello = result(server, "initialize", {"protocolVersion": "2025-11-25", "capabilities": {}})
        status, _, read = posted(server, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert (hello["protocolVersion"], (status, read)) == ("2025-11-25", (202, b""))
    told = cast("str", hello["instructions"])
    assert "- notes (" in told
    assert "hidden" not in told


def test_an_agent_connecting_while_the_collections_cannot_be_listed_is_told_so_and_how_to_search(
    home: Settings,
) -> None:
    damage_pages(home.home / CATALOG)
    with running(home) as server:
        hello = result(server, "initialize", {"protocolVersion": "2025-11-25", "capabilities": {}})
    told = cast("str", hello["instructions"])
    assert told.startswith(HOW)
    assert "\n\nsift cannot list the collections now: the catalog " in told
    assert "; fix: " in told.rpartition("\n\n")[2]


def test_an_agent_connecting_while_the_system_refuses_the_listing_is_told_so(home: Settings) -> None:
    def refused(_acting: Acting, _data: object) -> object:
        raise PermissionError(13, "Access is denied")

    with running(home, {**HANDLED, "status": Handled(refused, writes=False)}) as server:
        hello = result(server, "initialize", {"protocolVersion": "2025-11-25", "capabilities": {}})
    assert "\n\nsift cannot list the collections now: Access is denied; fix: " in cast("str", hello["instructions"])


def test_a_method_that_recurses_too_deep_is_not_answered_as_if_it_had_failed(home: Settings) -> None:
    def recursing(_acting: Acting, _data: object) -> object:
        raise RecursionError

    with running(home, {**HANDLED, "status": Handled(recursing, writes=False)}) as server:
        status, _, read = posted(server, {"jsonrpc": "2.0", "id": 7, "method": "initialize", "params": {}})
    assert (status, read) == (500, b"Internal Server Error")


def test_a_method_that_fails_answers_the_agent_s_message_and_is_logged(
    home: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(_acting: Acting, _data: object) -> object:
        raise RuntimeError("a bug")

    with running(home, {**HANDLED, "status": Handled(broken, writes=False)}) as server:
        status, headers, read = posted(server, {"jsonrpc": "2.0", "id": 7, "method": "initialize", "params": {}})
    assert (status, headers["content-type"]) == (200, "application/json")
    assert json.loads(read) == {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": -32603, "message": f"the home process failed: RuntimeError: a bug; fix: {FAILED_REMEDY}"},
    }
    assert "the message initialize failed" in caplog.text


def test_an_agent_searching_never_gets_an_answer_kept_for_people(home: Settings) -> None:
    with running(home) as server:
        status, lines = asked(server, "search", {"query": "dark matter"})
        people = [cast("Mapping[str, str]", hit)["id"] for hit in cast("list[object]", last(lines, "answer")["hits"])]
        called = result(
            server, "tools/call", {"name": "search", "arguments": {"query": "dark matter", "mode": "keyword"}}
        )
    hits = cast("Message", called["structuredContent"])["hits"]
    assert (status, sorted(people)) == (200, ["hidden/secret.md", "notes/halo.md"])
    assert [cast("Message", hit)["id"] for hit in cast("list[object]", hits)] == ["notes/halo.md"]


def test_an_agent_reindexes_a_file_it_wrote(home: Settings, notes: Path) -> None:
    write(notes / "comet.md", "# Comet\n\nA comet's tail.\n")
    with running(home) as server:
        called = result(server, "tools/call", {"name": "reindex", "arguments": {"files": ["notes/comet.md"]}})
        found = result(server, "tools/call", {"name": "search", "arguments": {"query": "comet", "mode": "keyword"}})
    assert cast("Message", called["structuredContent"])["files"] == [{"id": "notes/comet.md", "state": "added"}]
    hits = cast("list[Message]", cast("Message", found["structuredContent"])["hits"])
    assert [hit["id"] for hit in hits] == ["notes/comet.md"]


def test_a_tool_refused_tells_the_agent_how_to_fix_the_call(home: Settings) -> None:
    with running(home) as server:
        called = result(server, "tools/call", {"name": "search", "arguments": {"query": "x", "k": 0}})
    assert called["isError"] is True
    assert called["content"] == [{"type": "text", "text": "k: 0 is outside 1 to 10,000; fix: give 1 to 10,000"}]


@pytest.mark.parametrize(
    "headers,status",
    [
        ({"Origin": "https://attacker.example"}, 403),
        ({"Origin": "null"}, 403),
        ({"Origin": "http://[::1"}, 403),
        ({"Origin": "http://localhost:3000"}, 200),
        ({"MCP-Protocol-Version": "2099-01-01"}, 400),
        ({"MCP-Protocol-Version": "2025-06-18"}, 200),
        ({"Content-Type": "text/plain"}, 415),
        ({"Content-Type": "application/json; charset=utf-8"}, 200),
        ({"Host": "attacker.example"}, 400),
    ],
)
def test_requests_from_elsewhere_and_of_other_versions_are_refused(
    home: Settings, headers: Mapping[str, str], status: int
) -> None:
    with running(home) as server:
        answered = posted(server, {"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers)
    assert answered[0] == status


def test_what_is_not_one_posted_message_is_refused(home: Settings) -> None:
    with running(home) as server:
        listening, allowed, _ = posted(server, b"", method="GET")
        garbled, _, read = posted(server, b"{nope")
        long, _, _ = posted(server, b"[" * (1 << 21))
        past, _, beyond = posted(server, b'{"jsonrpc": "2.0", "id": 1e400, "method": "ping"}')
    assert (listening, allowed["allow"], garbled, long, past) == (405, "POST", 400, 413, 400)
    assert b"1e400, which is no JSON number" in beyond
    refusal = cast("Message", cast("Message", json.loads(read))["error"])
    assert refusal["code"] == -32700
    assert cast("str", refusal["message"]).startswith("the message is no JSON: Expecting property name")


def test_a_job_an_agent_left_runs_on_unwatched_and_is_listed(home: Settings) -> None:
    release, began = threading.Event(), threading.Event()

    def blocked(_acting: Acting, _data: object) -> object:
        began.set()
        assert release.wait(30)
        return None

    handled = {**HANDLED, "reindex": Handled(blocked, writes=True)}
    with running(home) as server:
        agents = AgentsRunning(server.serving, handled, wait=0.0)
        started = agents.job("reindex", {"files": ["notes/halo.md"]})
        assert began.wait(30)
        listed = agents.jobs()
        release.set()
    assert (started.answer, [(job.id, job.command, job.watched) for job in listed]) == (
        None,
        [(started.number, "reindex", False)],
    )


def test_a_job_that_itself_ran_out_of_time_has_failed_and_is_not_said_to_run_on(home: Settings) -> None:
    def timed_out(_acting: Acting, _data: object) -> object:
        raise TimeoutError("the share did not answer")

    with running(home) as server:
        agents = AgentsRunning(server.serving, {**HANDLED, "reindex": Handled(timed_out, writes=True)})
        with pytest.raises(TimeoutError, match="the share did not answer"):
            agents.job("reindex", {"files": ["notes/halo.md"]})
        assert agents.jobs() == ()


def _failing(error: Exception) -> Tool:
    def call(_running: ToolRunning, _arguments: Mapping[str, object]) -> Json:
        raise error

    return Tool("failing", "Failing", "fails", {"type": "object"}, call, read_only=True)


def test_a_tool_that_fails_tells_the_agent_what_failed(home: Settings, caplog: pytest.LogCaptureFixture) -> None:
    with running(home) as server:
        served = AgentsServed(AgentsRunning(server.serving, HANDLED))
        with pytest.raises(SystemFailureError):
            served.called(_failing(PermissionError(13, "Access is denied")), {})
        with pytest.raises(RecursionError):
            served.called(_failing(RecursionError("deep")), {})
        with pytest.raises(InternalError, match="the home process failed: RuntimeError: a bug"):
            served.called(_failing(RuntimeError("a bug")), {"query": "private words"})
    assert [record.getMessage() for record in caplog.records if record.levelname == "ERROR"] == [
        "the tool failing failed"
    ]


def test_a_job_with_a_bug_is_logged_and_its_failure_reaches_the_agent(
    home: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(_acting: Acting, _data: object) -> object:
        raise RuntimeError("a bug")

    with running(home) as server:
        agents = AgentsRunning(server.serving, {**HANDLED, "reindex": Handled(broken, writes=True)})
        with pytest.raises(RuntimeError, match="a bug"):
            agents.job("reindex", {"files": ["notes/halo.md"]})
        assert agents.jobs() == ()
    assert "reindex failed" in [record.getMessage() for record in caplog.records]


def test_an_agent_may_not_reindex_a_file_of_a_hidden_collection(home: Settings, tmp_path: Path) -> None:
    secret = str(tmp_path / "hidden" / "secret.md")
    with running(home) as server:
        called = result(server, "tools/call", {"name": "reindex", "arguments": {"files": [secret]}})
    assert called["isError"] is True
    assert called["content"] == [
        {"type": "text", "text": f"files: {secret} is in no collection; fix: {NO_COLLECTION_REMEDY}"}
    ]


def test_an_agent_searches_by_meaning_through_the_worker_the_home_process_sends_work_to(
    home: Settings, tmp_path: Path
) -> None:
    lent_settings = Settings(tmp_path / "worker", None, tmp_path, state=tmp_path / "worker-state")
    asking = {"name": "search", "arguments": {"query": "dark matter halo", "mode": "semantic", "k": 1}}
    with working(lent_settings, Hashed()) as worker:
        embed(home.home, EmbedCollections(), quiet, worker.remote())
        before = worker.requests()
        with running(dataclasses.replace(home, embed_remote=worker.address)) as server:
            called = result(server, "tools/call", asking)
        asked_there = worker.requests() - before
    hits = cast("list[Message]", cast("Message", called["structuredContent"])["hits"])
    assert ([hit["id"] for hit in hits], asked_there) == (["notes/halo.md"], 2)


def test_a_worker_the_home_process_cannot_reach_is_said_to_the_agent(home: Settings) -> None:
    port = nobody_listens()
    unreachable = dataclasses.replace(home, embed_remote=WorkerAddress(LOOPBACK, port, "s3cret", "p1n"))
    with running(unreachable) as server:
        called = result(server, "tools/call", {"name": "search", "arguments": {"query": "halo", "mode": "semantic"}})
    (said,) = cast("list[Message]", called["content"])
    text = cast("str", said["text"])
    assert (called["isError"], "s3cret" in text) == (True, False)
    assert text.startswith(f"embedding remote https://{LOOPBACK}:{port} is unreachable or failed: ")
