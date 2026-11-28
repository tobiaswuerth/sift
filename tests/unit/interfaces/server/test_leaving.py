import asyncio
import itertools
import signal
import threading

import pytest

from sift.api.serving import ServeHome
from sift.interfaces.server.activity import Activity
from sift.interfaces.server.endpoint import found, own_endpoint, written
from sift.interfaces.server.serving import (
    LEFT_IDLE,
    HomeServer,
    Serve,
    held,
    listing,
    on_a_selector,
    serve,
    watched_leaving,
)
from sift.settings import Settings
from tests.homes import quiet


def ticking() -> "itertools.count[float]":
    return itertools.count(0.0, 0.25)


def test_activity_is_idle_for_the_time_since_the_last_request_ended_and_never_while_one_runs() -> None:
    times = iter([0.0, 5.0, 9.0])
    activity = Activity(lambda: next(times))
    activity.began()
    assert activity.idle_for() == 0.0
    activity.ended()
    assert activity.idle_for() == 4.0


def test_a_home_process_told_to_leave_when_idle_stops_by_itself(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    clock = ticking()
    served = serve(
        settings, ServeHome(port=0, idle_exit=1), "2.0.0", quiet, Serve(clock=clock.__next__, idle_every=0.01)
    )
    assert (served.url.startswith("http://127.0.0.1:"), found(settings)) == (True, None)
    assert LEFT_IDLE.format(seconds=1) in [record.getMessage() for record in caplog.records]


def test_a_home_process_leaving_is_no_longer_found_from_the_moment_it_decides_to(settings: Settings) -> None:
    with held(settings, ServeHome(port=0), "2.0.0", quiet) as running, listing(running, written):
        assert own_endpoint(settings).exists()
        watched_leaving(running, 0, threading.Event(), 0.01)
        assert (running.server.should_exit, own_endpoint(settings).exists()) == (True, False)


def test_a_home_process_stopped_before_it_was_idle_long_enough_stops_its_watch(settings: Settings) -> None:
    def interrupted(_running: HomeServer) -> None:
        signal.raise_signal(signal.SIGINT)

    served = serve(settings, ServeHome(port=0, idle_exit=3_600), "2.0.0", quiet, Serve(interrupted, idle_every=0.01))
    assert (served.url.startswith("http://127.0.0.1:"), found(settings)) == (True, None)


async def _loop_running() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


def test_the_home_process_serves_on_a_selector_loop_whose_listening_socket_outlives_a_failed_accept() -> None:
    assert isinstance(on_a_selector(_loop_running()), asyncio.SelectorEventLoop)
