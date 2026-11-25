import io
import json
import os
import socket
import sys
import threading
from pathlib import Path
from typing import cast

import pytest

from sift.api.collections import AddCollection
from sift.app.collections import add
from sift.interfaces.server.endpoint import SERVER, SERVING, Endpoint, own_endpoint, written
from sift.interfaces.server.proxy import Pipes, Starting, in_the_background, proxied
from sift.settings import Settings
from sift.storage.writer import sole_writer
from tests.homes import indexed, quiet
from tests.serving import running
from tests.waits import joined

type Message = dict[str, object]

HELLO = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}
READY = {"jsonrpc": "2.0", "method": "notifications/initialized"}
LISTED = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
SEARCHED = {
    "jsonrpc": "2.0",
    "id": "three",
    "method": "tools/call",
    "params": {"name": "search", "arguments": {"query": "halo", "mode": "keyword"}},
}


def lines(*messages: object) -> io.BytesIO:
    return io.BytesIO(b"".join(m if isinstance(m, bytes) else json.dumps(m).encode() + b"\r\n" for m in messages))


def answers(written: io.BytesIO) -> dict[object, Message]:
    read = [cast("Message", json.loads(line)) for line in written.getvalue().splitlines()]
    return {each["id"]: each for each in read}


def test_each_message_goes_to_the_home_process_and_each_answer_back(settings: Settings, notes: Path) -> None:
    add(settings.home, AddCollection(str(notes)))
    indexed(settings.home)
    out = io.BytesIO()
    with running(settings):
        proxied(settings, quiet, Pipes(lines(HELLO, READY, b"\n", LISTED, SEARCHED), out))
    found = answers(out)
    assert sorted(found, key=str) == [1, 2, "three"]
    tools = cast("list[Message]", cast("Message", found[2]["result"])["tools"])
    assert [tool["name"] for tool in tools] == ["search", "get", "related", "collections", "reindex"]
    called = cast("Message", cast("Message", found["three"]["result"])["structuredContent"])
    assert [cast("Message", hit)["id"] for hit in cast("list[object]", called["hits"])] == ["notes/halo.md"]


def test_a_home_process_is_started_when_none_runs(settings: Settings) -> None:
    started, stop, out = list[Settings](), threading.Event(), io.BytesIO()

    def served(asked: Settings) -> None:
        with running(asked):
            assert stop.wait(30)

    def start(asked: Settings, idle: int) -> None:
        assert idle == 60
        started.append(asked)
        threads.append(threading.Thread(target=served, args=(asked,)))
        threads[-1].start()

    threads = list[threading.Thread]()
    try:
        proxied(settings, quiet, Pipes(lines(LISTED), out), Starting(start, idle=60))
    finally:
        stop.set()
        for thread in threads:
            joined(thread, 30)
    assert (started, list(answers(out))) == ([settings], [2])


def test_a_home_process_that_does_not_start_is_said_to_each_message_waiting(settings: Settings) -> None:
    out = io.BytesIO()
    proxied(settings, quiet, Pipes(lines(LISTED, READY), out), Starting(lambda _settings, _idle: None, patience=0.05))
    (refused,) = answers(out).values()
    assert refused == {
        "jsonrpc": "2.0",
        "id": 2,
        "error": {
            "code": -32603,
            "message": "the home process did not start within 0.05 s; fix: see its log in SIFT_HOME/logs",
        },
    }


def test_what_is_no_json_is_refused_without_an_id(settings: Settings) -> None:
    out = io.BytesIO()
    with running(settings):
        proxied(settings, quiet, Pipes(lines(b"{nope\n", LISTED), out))
    found = answers(out)
    assert set(found) == {2, None}
    assert cast("Message", found[None]["error"])["code"] == -32700


def test_a_home_process_that_stops_answering_is_said_to_each_message_waiting(settings: Settings) -> None:
    settings.home.mkdir(parents=True)
    with socket.create_server(("127.0.0.1", 0)) as taken:
        port = cast("tuple[str, int]", taken.getsockname())[1]
    written(own_endpoint(settings), Endpoint(port, os.getpid(), "t", "2.0.0"))
    out = io.BytesIO()
    with sole_writer(settings.home / SERVER, SERVING):
        proxied(settings, quiet, Pipes(lines(LISTED, b'{"id": 5, oops\n'), out))
    (refused,) = answers(out).values()
    error = cast("Message", refused["error"])
    assert (refused["id"], error["code"]) == (2, -32603)
    assert cast("str", error["message"]).startswith("the home process does not answer: ")


def connections_waiting(listening: socket.socket) -> int:
    listening.settimeout(0)
    found = 0
    while True:
        try:
            listening.accept()[0].close()
        except BlockingIOError:
            return found
        found += 1


def test_a_home_process_that_stays_silent_is_asked_once_and_said_to_the_message_waiting(settings: Settings) -> None:
    settings.home.mkdir(parents=True)
    out = io.BytesIO()
    with socket.create_server(("127.0.0.1", 0)) as silent:
        port = cast("tuple[str, int]", silent.getsockname())[1]
        written(own_endpoint(settings), Endpoint(port, os.getpid(), "t", "2.0.0"))
        with sole_writer(settings.home / SERVER, SERVING):
            proxied(settings, quiet, Pipes(lines(LISTED), out), Starting(quiet=0.2))
        asked = connections_waiting(silent)
    (refused,) = answers(out).values()
    said = cast("str", cast("Message", refused["error"])["message"])
    assert (asked, refused["id"], said.startswith("the home process does not answer: ")) == (1, 2, True)


WINDOW_SAID = (
    "import ctypes, sys; k, u = ctypes.windll.kernel32, ctypes.windll.user32; w = k.GetConsoleWindow(); "
    "open(sys.argv[1], 'w').write(f'{w != 0} {bool(u.IsWindowVisible(w))} {bool(u.IsIconic(w))}')"
)


@pytest.mark.skipif(sys.platform != "win32", reason="console windows are Windows' own")
def test_a_process_started_in_the_background_shows_its_window_minimized(tmp_path: Path) -> None:
    said = tmp_path / "window.txt"
    with in_the_background([sys.executable, "-c", WINDOW_SAID, str(said)], tmp_path) as running:
        assert running.wait(30) == 0
    assert said.read_text(encoding="utf-8") == "True True True"
