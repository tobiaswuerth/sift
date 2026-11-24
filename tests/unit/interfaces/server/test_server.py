import http.client
import json
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, Problem, StoreUnavailableError
from sift.api.events import Event, Listening
from sift.api.serving import Served, ServeHome
from sift.app.embedding import embed
from sift.interfaces.acts.acting import Acting
from sift.interfaces.cli.main import ENTRIES
from sift.interfaces.server.app import WAITING
from sift.interfaces.server.endpoint import ENDPOINT, SERVER, SERVING, Endpoint, found, lock_of
from sift.interfaces.server.handled import HANDLED, Handled
from sift.interfaces.server.serving import HomeServer, held, serve
from sift.settings import Settings
from sift.storage.writer import held_by, sole_writer
from tests.files import write
from tests.models import Hashed
from tests.serving import running

type Line = dict[str, object]


def asked(
    server: HomeServer, name: str, body: object, *, bearer: str | None = None, host: str | None = None
) -> tuple[int, list[Line]]:
    connection = http.client.HTTPConnection("127.0.0.1", server.endpoint.port, timeout=60)
    headers = {"Authorization": f"Bearer {server.endpoint.token if bearer is None else bearer}"}
    if host is not None:
        headers["Host"] = host
    sent = body if isinstance(body, bytes) else json.dumps(body).encode()
    try:
        connection.request("POST", f"/api/{name}", sent, headers)
        response = connection.getresponse()
        status, read = response.status, response.read()
    finally:
        connection.close()
    if not read.startswith(b"{"):
        return status, [{"text": read.decode()}]
    return status, [cast("Line", json.loads(line)) for line in read.split(b"\n") if line]


def last(lines: list[Line], key: str) -> Line:
    return cast("Line", lines[-1][key])


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(home=tmp_path / "home", user=None, cwd=tmp_path)


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    write(tmp_path / "notes" / "halo.md", "# Halo\n\nDark matter halo models.\n")
    write(tmp_path / "notes" / "brush.md", "# Brush\n\nA robot plans brush strokes.\n")
    return tmp_path / "notes"


def test_the_home_process_answers_every_command_the_command_line_has_but_serve() -> None:
    assert set(HANDLED) == set(ENTRIES) - {"serve"}


def test_the_home_process_runs_commands_streaming_their_events_then_their_answer(
    settings: Settings, notes: Path
) -> None:
    with running(settings) as server:
        assert found(settings.home) == server.endpoint
        status, added = asked(server, "add", {"folder": str(notes)})
        assert (status, last(added, "answer")["state"]) == (200, "added")
        _, indexed = asked(server, "index", {})
        events = [line["event"] for line in indexed[:-1]]
        assert (events[0], events[-1]) == ("CollectionStarted", "Finished")
        (collection,) = cast("list[Line]", last(indexed, "answer")["collections"])
        assert cast("Line", collection["outcome"])["added"] == 2
        _, searched = asked(server, "search", {"query": "robot"})
        assert [hit["id"] for hit in cast("list[Line]", last(searched, "answer")["hits"])] == ["notes/brush.md"]
    assert (found(settings.home), (settings.home / ENDPOINT).exists()) == (None, False)


def test_searches_through_the_home_process_keep_a_store_s_codes(settings: Settings, notes: Path) -> None:
    with running(settings) as server:
        asked(server, "add", {"folder": str(notes)})
        asked(server, "index", {})
        embed(settings.home, EmbedCollections(), lambda _: None, Hashed())
        first = asked(server, "similar", {"id": "notes/halo.md"})[1]
        again = asked(server, "similar", {"id": "notes/halo.md"})[1]
        assert first == again
        assert [store.name for store, _ in server.serving.codes.held] == ["notes.sqlite"]


def decoded_in(server: HomeServer) -> list[str]:
    return sorted({store.name for store, *_ in server.serving.decoded.entries})


def test_codes_and_keywords_of_a_collection_deactivated_or_removed_are_let_go(
    settings: Settings, notes: Path, tmp_path: Path
) -> None:
    write(tmp_path / "papers" / "shaping.md", "# Shaping\n\nReward shaping speeds up learning.\n")
    with running(settings) as server:
        for folder in (notes, tmp_path / "papers"):
            asked(server, "add", {"folder": str(folder)})
        asked(server, "index", {})
        embed(settings.home, EmbedCollections(), lambda _: None, Hashed())
        asked(server, "similar", {"id": "notes/halo.md"})
        asked(server, "similar", {"id": "papers/shaping.md"})
        asked(server, "search", {"query": "halo OR shaping"})
        assert sorted(store.name for store, _ in server.serving.codes.held) == ["notes.sqlite", "papers.sqlite"]
        assert decoded_in(server) == ["notes.sqlite", "papers.sqlite"]
        asked(server, "deactivate", {"names": ["notes"]})
        assert [store.name for store, _ in server.serving.codes.held] == ["papers.sqlite"]
        assert decoded_in(server) == ["papers.sqlite"]
        asked(server, "rm", {"name": "papers"})
        assert (list(server.serving.codes.held), decoded_in(server)) == ([], [])


def test_a_call_without_its_token_from_another_host_or_to_no_command_is_refused(settings: Settings) -> None:
    with running(settings) as server:
        assert [asked(server, "status", {}, bearer=given)[0] for given in ("", "wrong")] == [401, 401]
        refused = asked(server, "status", {}, bearer="wrong")[1][0]
        assert cast("Line", refused["error"])["code"] == Problem.UNAUTHORIZED.value
        assert asked(server, "status", {}, host="sift.example.com")[0] == 400
        assert asked(server, "status", {}, host=f"localhost:{server.endpoint.port}")[0] == 200
        status, unknown = asked(server, "frobnicate", {})
        assert (status, cast("Line", unknown[0]["error"])["code"]) == (404, Problem.NOT_FOUND.value)


def test_a_request_that_is_no_json_too_long_or_wrong_is_said_to_be(settings: Settings) -> None:
    with running(settings) as server:
        assert asked(server, "status", b"{not json")[0] == 400
        assert asked(server, "status", b'{"x": "' + b"x" * (1 << 20) + b'"}')[0] == 413
        status, wrong = asked(server, "search", {"query": "x", "k": 0})
        refused = last(wrong, "error")
        assert (status, refused["code"], refused["field"]) == (200, Problem.INVALID_REQUEST.value, "k")


def broken(_acting: Acting, _data: object) -> object:
    raise ValueError("a bug")


def test_a_command_that_fails_unforeseen_is_said_and_the_home_process_goes_on(settings: Settings) -> None:
    with running(settings, {**HANDLED, "status": Handled(broken, writes=False)}) as server:
        failed = last(asked(server, "status", {})[1], "error")
        assert (failed["code"], failed["message"]) == (
            Problem.INTERNAL.value,
            "the home process failed: ValueError: a bug",
        )
        assert last(asked(server, "unresolved", {})[1], "answer") == {"targets": [], "unsearched": []}


def recursing(_acting: Acting, _data: object) -> object:
    raise RecursionError("maximum recursion depth exceeded")


def test_a_command_that_recurses_too_deep_ends_its_answer_unfinished(settings: Settings) -> None:
    with running(settings, {**HANDLED, "status": Handled(recursing, writes=False)}) as server:
        with pytest.raises(http.client.IncompleteRead):
            asked(server, "status", {})
        assert last(asked(server, "unresolved", {})[1], "answer") == {"targets": [], "unsearched": []}


@contextmanager
def streaming(server: HomeServer, name: str) -> Generator[http.client.HTTPResponse]:
    connection = http.client.HTTPConnection("127.0.0.1", server.endpoint.port, timeout=60)
    try:
        connection.request("POST", f"/api/{name}", b"{}", {"Authorization": f"Bearer {server.endpoint.token}"})
        with connection.getresponse() as response:
            yield response
    finally:
        connection.close()


def test_a_quiet_command_s_answer_is_kept_alive_by_a_line_while_nothing_else_is_said(settings: Settings) -> None:
    release = threading.Event()

    def quiet(_acting: Acting, _data: object) -> object:
        release.wait(30)
        return Served("quiet")

    with (
        running(settings, {**HANDLED, "status": Handled(quiet, writes=False)}, alive_every=0.01) as server,
        streaming(server, "status") as response,
    ):
        first = response.readline()
        release.set()
        rest = response.read()
        kind = response.getheader("Content-Type")
    assert (kind, first) == ("application/x-ndjson; charset=utf-8", b'{"alive": true}\n')
    assert rest.split(b"\n")[-2:] == [b'{"answer": {"url": "quiet"}}', b""]


def test_a_write_sent_while_another_writes_is_told_it_waits_then_runs(settings: Settings) -> None:
    started, release = threading.Event(), threading.Event()

    def writing(_acting: Acting, _data: object) -> object:
        started.set()
        release.wait(30)
        return Served("written")

    with running(settings, {**HANDLED, "add": Handled(writing, writes=True)}) as server:
        first = threading.Thread(target=asked, args=(server, "add", {}), daemon=True)
        first.start()
        started.wait(30)
        with streaming(server, "add") as response:
            told = cast("Line", json.loads(response.readline()))
            release.set()
            rest = response.read()
        first.join(30)
    assert told == {"event": "Noted", "data": {"text": WAITING, "remedy": None}}
    assert rest.endswith(b'{"answer": {"url": "written"}}\n')


def interrupted(_event: Event) -> None:
    raise KeyboardInterrupt


def test_ctrl_c_as_the_home_process_says_where_it_listens_stops_it_as_at_any_other_time(settings: Settings) -> None:
    ran = list[HomeServer]()
    served = serve(settings, ServeHome(port=0), "2.0.0", interrupted, ran.append)
    assert (ran, served.url.startswith("http://127.0.0.1:"), found(settings.home)) == ([], True, None)


def test_serve_says_where_it_listens_serves_and_leaves_nothing_behind(settings: Settings) -> None:
    said, seen = list[Event](), list[Endpoint | None]()
    served = serve(settings, ServeHome(port=0), "2.0.0", said.append, lambda _: seen.append(found(settings.home)))
    (endpoint,) = seen
    assert endpoint is not None
    url = f"http://127.0.0.1:{endpoint.port}"
    assert (said, served) == ([Listening(url, str(settings.home))], Served(url))
    assert (found(settings.home), (settings.home / ENDPOINT).exists()) == (None, False)


def test_a_server_whose_endpoint_is_not_written_yet_is_none_to_find(settings: Settings) -> None:
    with sole_writer(settings.home / SERVER, SERVING):
        assert held_by(lock_of(settings.home)) is not None
        assert found(settings.home) is None


def test_a_second_home_process_is_refused_naming_the_first(settings: Settings) -> None:
    with running(settings) as server:
        with (
            pytest.raises(StoreUnavailableError, match=f"on port {server.endpoint.port}") as refused,
            held(settings, ServeHome(port=0), "2.0.0"),
        ):
            pass
        assert refused.value.code == Problem.BUSY


def test_a_port_that_is_taken_is_said_to_be(settings: Settings) -> None:
    other = Settings(settings.home.parent / "other", settings.user, settings.cwd)
    with running(settings) as server:
        taken = ServeHome(server.endpoint.port)
        with (
            pytest.raises(InvalidRequestError, match=f"port {taken.port} cannot be had") as refused,
            held(other, taken, "2.0.0"),
        ):
            pass
        assert refused.value.field == "port"
