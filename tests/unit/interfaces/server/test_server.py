import http.client
import json
import logging
import os
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
from sift.interfaces.server.app import LOGGER, MOST_BODY, WAITING
from sift.interfaces.server.endpoint import (
    ENDPOINT,
    LOOK_EVERY_SECONDS,
    SERVER,
    SERVING,
    Endpoint,
    found,
    lock_of,
    written,
)
from sift.interfaces.server.gate import WAITING_FOR_ALONE, WAITING_FOR_READS
from sift.interfaces.server.handled import HANDLED, Handled
from sift.interfaces.server.serving import RESTED, HomeServer, held, serve, watched_idle
from sift.settings import Settings
from sift.storage.catalog import known_collections
from sift.storage.store import read_store
from sift.storage.writer import held_by, sole_writer
from tests.files import write
from tests.models import Hashed
from tests.serving import Line, asked, last, running


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
        wrong = ("Bearer ", "Bearer wrong")
        assert [asked(server, "status", {}, authorization=given)[0] for given in wrong] == [401, 401]
        refused = asked(server, "status", {}, authorization="Bearer wrong")[1][0]
        assert cast("Line", refused["error"])["code"] == Problem.UNAUTHORIZED.value
        assert asked(server, "status", {}, host="sift.example.com")[0] == 400
        assert asked(server, "status", {}, host=f"localhost:{server.endpoint.port}")[0] == 200
        status, unknown = asked(server, "frobnicate", {})
        assert (status, cast("Line", unknown[0]["error"])["code"]) == (404, Problem.NOT_FOUND.value)


@pytest.mark.parametrize("authorization", ["", "Basic dGhlOnRva2Vu", "bearer {token}", "Bearer  {token}"])
@pytest.mark.parametrize("name", ["status", "rm"])
def test_a_call_without_the_token_in_its_one_form_is_refused(settings: Settings, authorization: str, name: str) -> None:
    with running(settings) as server:
        given = authorization.format(token=server.endpoint.token)
        status, lines = asked(server, name, {"name": "notes"}, authorization=given)
    assert (status, cast("Line", lines[0]["error"])["code"]) == (401, Problem.UNAUTHORIZED.value)


def test_a_request_of_exactly_the_most_a_request_holds_is_read(settings: Settings) -> None:
    with running(settings) as server:
        assert asked(server, "status", b"{}" + b" " * (MOST_BODY - 2))[0] == 200
        assert asked(server, "status", b"{}" + b" " * (MOST_BODY - 1))[0] == 413


@pytest.mark.parametrize(
    "body, said",
    [
        (b"[" * 200_000 + b"]" * 200_000, "the request is nested more than 64 deep"),
        (b'{"query": "x", "budget": NaN}', "the request holds NaN, which is no JSON number"),
        (b'{"query": "x", "budget": -Infinity}', "the request holds -Infinity, which is no JSON number"),
        (b'{"query": "x", "query": "y"}', "the request gives the key 'query' twice"),
        (b'{"query": "\xff"}', "the request is no JSON"),
    ],
)
def test_json_too_deep_not_finite_with_a_key_twice_or_not_utf_8_is_refused(
    settings: Settings, body: bytes, said: str
) -> None:
    with running(settings) as server:
        status, lines = asked(server, "search", body)
    refused = cast("Line", lines[0]["error"])
    assert (status, refused["code"], refused["message"]) == (400, Problem.INVALID_REQUEST.value, said)


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
def streaming(server: HomeServer, name: str, body: object = None) -> Generator[http.client.HTTPResponse]:
    connection = http.client.HTTPConnection("127.0.0.1", server.endpoint.port, timeout=60)
    sent = json.dumps({} if body is None else body).encode()
    try:
        connection.request("POST", f"/api/{name}", sent, {"Authorization": f"Bearer {server.endpoint.token}"})
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


def answer_in(read: bytes) -> Line:
    (answer,) = [
        cast("Line", json.loads(line))["answer"] for line in read.splitlines() if line.startswith(b'{"answer"')
    ]
    return cast("Line", answer)


def test_rm_waits_for_searches_of_its_store_and_deletes_it(settings: Settings, notes: Path) -> None:
    started, release = threading.Event(), threading.Event()

    def searching(acting: Acting, _data: object) -> object:
        (one,) = known_collections(acting.settings.home)
        with read_store(one.store, one.id):
            started.set()
            release.wait(30)
        return Served("searched")

    with running(settings, {**HANDLED, "search": Handled(searching, writes=False)}) as server:
        asked(server, "add", {"folder": str(notes)})
        asked(server, "index", {})
        reader = threading.Thread(target=asked, args=(server, "search", {}), daemon=True)
        reader.start()
        started.wait(30)
        with streaming(server, "rm", {"name": "notes"}) as response:
            told = cast("Line", json.loads(response.readline()))
            release.set()
            rest = response.read()
        reader.join(30)
        _, added = asked(server, "add", {"folder": str(notes)})
    assert told == {"event": "Noted", "data": {"text": WAITING_FOR_READS, "remedy": None}}
    assert answer_in(rest)["left"] == []
    assert last(added, "answer")["state"] == "added"


def test_a_search_sent_while_a_store_is_removed_waits_for_it_then_runs(settings: Settings) -> None:
    started, release = threading.Event(), threading.Event()

    def removing(_acting: Acting, _data: object) -> object:
        started.set()
        release.wait(30)
        return Served("removed")

    with running(settings, {**HANDLED, "rm": Handled(removing, writes=True, alone=True)}) as server:
        remover = threading.Thread(target=asked, args=(server, "rm", {"name": "notes"}), daemon=True)
        remover.start()
        started.wait(30)
        with streaming(server, "status") as response:
            told = cast("Line", json.loads(response.readline()))
            release.set()
            rest = response.read()
        remover.join(30)
    assert told == {"event": "Noted", "data": {"text": WAITING_FOR_ALONE, "remedy": None}}
    assert answer_in(rest)["collections"] == []


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


def test_a_home_process_whose_endpoint_is_not_written_yet_is_starting_and_said_to_be(settings: Settings) -> None:
    with sole_writer(settings.home / SERVER, SERVING):
        assert held_by(lock_of(settings.home)) is not None
        with pytest.raises(StoreUnavailableError, match=f"is starting \\(process {os.getpid()}\\)") as refused:
            found(settings.home, patience=LOOK_EVERY_SECONDS)
    assert (refused.value.code, refused.value.remedy) == (Problem.BUSY, "ask again in a moment")


def test_the_endpoint_of_another_process_is_not_the_home_process_that_holds_the_home(settings: Settings) -> None:
    settings.home.mkdir(parents=True)
    written(settings.home, Endpoint(9, os.getpid() + 1, "t", "2.0.0"))
    assert found(settings.home) is None
    with sole_writer(settings.home / SERVER, SERVING), pytest.raises(StoreUnavailableError, match="is starting"):
        found(settings.home, patience=LOOK_EVERY_SECONDS * 2)


def test_a_home_process_takes_the_place_of_an_endpoint_left_by_one_that_was_killed(settings: Settings) -> None:
    settings.home.mkdir(parents=True)
    written(settings.home, Endpoint(9, os.getpid(), "t", "2.0.0"))
    with running(settings) as server:
        assert found(settings.home) == server.endpoint != Endpoint(9, os.getpid(), "t", "2.0.0")


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


def test_the_home_process_lets_idle_models_go_and_says_so(caplog: pytest.LogCaptureFixture) -> None:
    stop, asked = threading.Event(), list[bool]()

    def release() -> bool:
        asked.append(len(asked) == 0)
        if len(asked) == 2:
            stop.set()
        return asked[-1]

    with caplog.at_level(logging.INFO, LOGGER):
        watched_idle(release, stop, 0.001)
    assert (asked, [record.getMessage() for record in caplog.records]) == ([True, False], [RESTED])
