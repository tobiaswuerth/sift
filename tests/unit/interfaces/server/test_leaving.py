import itertools

import pytest

from sift.api.serving import ServeHome
from sift.interfaces.server.activity import Activity
from sift.interfaces.server.endpoint import found
from sift.interfaces.server.serving import LEFT_IDLE, HomeServer, Serve, serve
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


def test_a_home_process_stopped_before_it_was_idle_long_enough_stops_its_watch(settings: Settings) -> None:
    def interrupted(_running: HomeServer) -> None:
        raise KeyboardInterrupt

    served = serve(settings, ServeHome(port=0, idle_exit=3_600), "2.0.0", quiet, Serve(interrupted, idle_every=0.01))
    assert (served.url.startswith("http://127.0.0.1:"), found(settings)) == (True, None)
