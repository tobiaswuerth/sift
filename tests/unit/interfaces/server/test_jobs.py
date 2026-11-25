import contextlib
import http.client
import io
import json
import signal
import threading
from collections.abc import Callable
from typing import cast

from sift.api.errors import JobStoppedError
from sift.api.events import Advanced, CollectionStarted, Event, Noted
from sift.api.serving import Served, ServeHome
from sift.interfaces.acts.acting import NO_HOME_PROCESS, Acting
from sift.interfaces.cli.main import Host, run
from sift.interfaces.server.endpoint import found, own_endpoint
from sift.interfaces.server.handled import HANDLED, Handled
from sift.interfaces.server.serving import STOPPING, HomeServer, Serve, held, serve
from sift.settings import Settings
from tests.serving import Line, asked, last, running, streaming
from tests.waits import joined, paused, until, waited

STEPS = 1_000_000


def endless(started: threading.Event, ended: list[str]) -> Callable[[Acting, object], object]:
    def indexing(acting: Acting, _data: object) -> object:
        acting.report(CollectionStarted("notes", "folder"))
        started.set()
        try:
            for done in range(1, STEPS):
                acting.report(Advanced(done))
                paused(0.01)
        except JobStoppedError:
            ended.append("stopped")
            raise
        return Served("never")

    return indexing


def jobs_of(server: HomeServer, body: object = None) -> Line:
    return last(asked(server, "jobs", {} if body is None else body)[1], "answer")


def listed(server: HomeServer) -> list[Line]:
    return cast("list[Line]", jobs_of(server)["jobs"])


def test_a_job_left_by_its_client_is_listed_and_can_be_stopped(settings: Settings) -> None:
    started, ended = threading.Event(), list[str]()
    with running(settings, {**HANDLED, "index": Handled(endless(started, ended), writes=True)}) as server:
        with streaming(server, "index") as response:
            _ = response.readline()
        assert started.wait(30)
        assert until(lambda: listed(server)[0]["watched"] is False)
        (job,) = listed(server)
        assert (job["command"], job["running"], job["collection"]) == ("index", True, "notes")
        assert jobs_of(server, {"stop": job["id"]})["stopped"] == job["id"]
        assert until(lambda: listed(server) == [])
    assert ended == ["stopped"]


def test_a_job_stopped_while_it_waits_ends_before_it_begins_and_its_client_is_told(settings: Settings) -> None:
    started, release, ran = threading.Event(), threading.Event(), list[str]()

    def writing(_acting: Acting, _data: object) -> object:
        ran.append("ran")
        started.set()
        waited(release, 30)
        return Served("written")

    with running(settings, {**HANDLED, "add": Handled(writing, writes=True)}) as server:
        first = threading.Thread(target=asked, args=(server, "add", {}), daemon=True)
        first.start()
        assert started.wait(30)
        with streaming(server, "add") as response:
            _ = response.readline()
            assert jobs_of(server, {"stop": 2})["stopped"] == 2
            release.set()
            rest = response.read()
        joined(first, 30)
    (error,) = [cast("Line", json.loads(line))["error"] for line in rest.splitlines() if line.startswith(b'{"error"')]
    assert (cast("Line", error)["kind"], ran) == ("stopped", ["ran"])


def test_searches_are_answered_while_many_writes_wait(settings: Settings) -> None:
    started, release = threading.Event(), threading.Event()

    def writing(_acting: Acting, _data: object) -> object:
        started.set()
        waited(release, 120)
        return Served("written")

    with running(settings, {**HANDLED, "add": Handled(writing, writes=True)}) as server:
        writes = [threading.Thread(target=asked, args=(server, "add", {}), daemon=True) for _ in range(41)]
        writes[0].start()
        assert started.wait(30)
        for each in writes[1:]:
            each.start()
        assert until(lambda: len(listed(server)) == 41)
        status, said = asked(server, "status", {})
        release.set()
        for each in writes:
            joined(each, 30)
    assert (status, last(said, "answer")["collections"]) == (200, [])


def test_ctrl_c_with_a_job_running_stops_the_job_and_says_so(settings: Settings) -> None:
    started, ended, said = threading.Event(), list[str](), list[Event]()

    def interrupted_while_indexing(running: HomeServer) -> None:
        assert until(lambda: running.server.started)
        threading.Thread(target=asked, args=(running, "index", {}), daemon=True).start()
        assert started.wait(30)
        signal.raise_signal(signal.SIGINT)

    def served(running: HomeServer) -> None:
        helper = threading.Thread(target=interrupted_while_indexing, args=(running,), daemon=True)
        helper.start()
        try:
            running.server.run(sockets=[running.socket])
        finally:
            joined(helper, 30)

    handled = {**HANDLED, "index": Handled(endless(started, ended), writes=True)}
    serve(settings, ServeHome(port=0), "2.0.0", said.append, Serve(served, handled))
    assert (ended, found(settings)) == (["stopped"], None)
    assert [event for event in said if isinstance(event, Noted)] == [Noted(STOPPING.format(jobs="sift index (job 1)"))]


def ran(settings: Settings, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = run(argv, settings, out, err, Host(clock=lambda: 0.0))
    return code, out.getvalue(), err.getvalue()


def test_without_a_home_process_no_job_is_listed_and_none_can_be_stopped(settings: Settings) -> None:
    assert ran(settings, "jobs") == (0, "", " none  no job runs or waits in the background\n")
    code, _, err = ran(settings, "jobs", "--stop", "3")
    assert (code, err.splitlines()[0]) == (1, f"error  there is no job 3: {NO_HOME_PROCESS}")


def test_the_jobs_of_the_home_process_are_listed_and_one_is_stopped_from_the_command_line(settings: Settings) -> None:
    started, ended = threading.Event(), list[str]()
    with running(settings, {**HANDLED, "index": Handled(endless(started, ended), writes=True)}) as server:
        threading.Thread(target=asked, args=(server, "index", {}), daemon=True).start()
        assert started.wait(30)
        code, out, _ = ran(settings, "jobs")
        assert (code, out.splitlines()[0].split()) == (
            0,
            ["job", "command", "state", "seconds", "collection", "done", "watched"],
        )
        assert out.splitlines()[1].split()[:3] == ["1", "index", "runs"]
        code, _, err = ran(settings, "jobs", "--stop", "1")
        assert (code, "job 1 stops at its next step" in err) == (0, True)
        assert until(lambda: listed(server) == [])
        refused = asked(server, "jobs", {"stop": 99})[1]
        assert cast("Line", refused[-1]["error"])["message"] == "there is no job 99"
    assert ended == ["stopped"]


def test_a_second_ctrl_c_exits_at_once(settings: Settings) -> None:
    with running(settings) as server:
        server.server.handle_exit(signal.SIGINT, None)
        server.server.handle_exit(signal.SIGINT, None)
        gone = not own_endpoint(settings).exists()
        assert (server.server.should_exit, server.server.force_exit, gone) == (True, True, True)


def test_a_write_that_ends_after_its_client_was_let_go_ends_quietly(settings: Settings) -> None:
    started, release, ended = threading.Event(), threading.Event(), list[str]()

    def writing(_acting: Acting, _data: object) -> object:
        started.set()
        waited(release, 30)
        ended.append("ended")
        return Served("written")

    def gone(server: HomeServer) -> None:
        with contextlib.suppress(http.client.HTTPException, OSError):
            asked(server, "add", {})

    how = Serve(handled={**HANDLED, "add": Handled(writing, writes=True)}, grace_seconds=0)
    with held(settings, ServeHome(port=0), "2.0.0", list[Event]().append, how) as server:
        thread = threading.Thread(target=server.server.run, kwargs={"sockets": [server.socket]}, daemon=True)
        thread.start()
        assert until(lambda: server.server.started)
        client = threading.Thread(target=gone, args=(server,), daemon=True)
        client.start()
        assert started.wait(30)
        server.server.handle_exit(signal.SIGINT, None)
        server.server.handle_exit(signal.SIGINT, None)
        joined(thread, 30)
        joined(client, 30)
        release.set()
    assert ended == ["ended"]


def test_sift_serve_ended_as_a_service_is_ended_stops_as_by_ctrl_c(settings: Settings) -> None:
    def terminated_once_started(running: HomeServer) -> None:
        assert until(lambda: running.server.started)
        signal.raise_signal(signal.SIGTERM)

    def served(running: HomeServer) -> None:
        helper = threading.Thread(target=terminated_once_started, args=(running,), daemon=True)
        helper.start()
        try:
            running.server.run(sockets=[running.socket])
        finally:
            joined(helper, 30)

    before = signal.getsignal(signal.SIGTERM)
    served_url = serve(settings, ServeHome(port=0), "2.0.0", list[Event]().append, Serve(served)).url
    assert (served_url.startswith("http://127.0.0.1:"), found(settings)) == (True, None)
    assert signal.getsignal(signal.SIGTERM) is before
