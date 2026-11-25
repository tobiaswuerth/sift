import dataclasses
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from sift.api.errors import InvalidRequestError, Problem, StoreUnavailableError
from sift.api.events import Event
from sift.api.serving import ServeHome
from sift.interfaces.server.endpoint import (
    LOOK_EVERY_SECONDS,
    NO_OWN_FOLDER,
    NO_OWN_FOLDER_REMEDY,
    SERVER,
    SERVING,
    STARTING,
    STARTING_REMEDY,
    Endpoint,
    found,
    lock_of,
    own_endpoint,
    written,
)
from sift.interfaces.server.serving import held
from sift.settings import Settings
from sift.storage.database import PRIVATE_FILE, PRIVATE_FOLDER
from sift.storage.writer import held_by, sole_writer
from tests.serving import running


def test_a_home_process_whose_endpoint_is_not_written_yet_is_starting_and_said_to_be(settings: Settings) -> None:
    with sole_writer(settings.home / SERVER, SERVING):
        assert held_by(lock_of(settings.home)) is not None
        with pytest.raises(StoreUnavailableError) as refused:
            found(settings, patience=LOOK_EVERY_SECONDS)
    assert (str(refused.value), refused.value.code) == (STARTING.format(pid=os.getpid()), Problem.BUSY)
    assert refused.value.remedy == STARTING_REMEDY


def test_the_home_process_keeps_its_token_in_its_user_s_own_folder_and_none_of_it_in_the_home(
    settings: Settings,
) -> None:
    with running(settings) as server:
        token, listed = server.endpoint.token, own_endpoint(settings)
        assert (listed.parent.parent, json.loads(listed.read_bytes())["token"]) == (settings.state, token)
        assert found(dataclasses.replace(settings, home=settings.home / ".." / settings.home.name)) == server.endpoint
    held = [path for path in settings.home.rglob("*") if path.is_file() and token.encode() in path.read_bytes()]
    assert (held, listed.exists()) == ([], False)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_the_home_process_s_endpoint_is_readable_by_its_user_alone(settings: Settings) -> None:
    with running(settings):
        listed = own_endpoint(settings)
        modes = (stat.S_IMODE(listed.stat().st_mode), stat.S_IMODE(listed.parent.stat().st_mode))
    assert modes == (PRIVATE_FILE, PRIVATE_FOLDER)


def test_a_home_process_that_serves_another_user_is_said_to_be_one(settings: Settings, tmp_path: Path) -> None:
    others = (dataclasses.replace(settings, state=tmp_path / "another"), dataclasses.replace(settings, state=None))
    with running(settings):
        for another in others:
            with pytest.raises(StoreUnavailableError, match="or serves another user") as refused:
                found(another, patience=LOOK_EVERY_SECONDS)
            assert refused.value.code == Problem.BUSY
    assert [found(another) for another in others] == [None, None]


def test_serve_without_a_folder_of_the_user_s_own_is_refused_saying_what_to_set(settings: Settings) -> None:
    with (
        pytest.raises(InvalidRequestError, match=NO_OWN_FOLDER) as refused,
        held(dataclasses.replace(settings, state=None), ServeHome(port=0), "2.0.0", list[Event]().append),
    ):
        pass
    assert refused.value.remedy == NO_OWN_FOLDER_REMEDY


@pytest.mark.parametrize("kept", [b"{nope", b"[]", b'{"port": "x"}', b"\xff", b""])
def test_a_home_process_whose_endpoint_cannot_be_read_is_refused_saying_what_to_do(
    settings: Settings, kept: bytes
) -> None:
    listed = own_endpoint(settings)
    listed.parent.mkdir(parents=True)
    listed.write_bytes(kept)
    assert found(settings) is None
    with sole_writer(settings.home / SERVER, SERVING), pytest.raises(StoreUnavailableError) as refused:
        found(settings, patience=LOOK_EVERY_SECONDS)
    said = f"the file in which the home process (process {os.getpid()}) says where it listens cannot be read: {listed}"
    assert (refused.value.message, refused.value.remedy) == (said, "stop that process and start sift serve again")


def test_the_endpoint_of_another_process_is_not_the_home_process_that_holds_the_home(settings: Settings) -> None:
    settings.home.mkdir(parents=True)
    written(own_endpoint(settings), Endpoint(9, os.getpid() + 1, "t", "2.0.0"))
    assert found(settings) is None
    with sole_writer(settings.home / SERVER, SERVING), pytest.raises(StoreUnavailableError, match="is starting"):
        found(settings, patience=LOOK_EVERY_SECONDS * 2)


def test_a_home_process_takes_the_place_of_an_endpoint_left_by_one_that_was_killed(settings: Settings) -> None:
    settings.home.mkdir(parents=True)
    written(own_endpoint(settings), Endpoint(9, os.getpid(), "t", "2.0.0"))
    with running(settings) as server:
        assert found(settings) == server.endpoint != Endpoint(9, os.getpid(), "t", "2.0.0")


def test_a_second_home_process_is_refused_naming_the_first(settings: Settings) -> None:
    with running(settings) as server:
        with (
            pytest.raises(StoreUnavailableError, match=f"on port {server.endpoint.port}") as refused,
            held(settings, ServeHome(port=0), "2.0.0", list[Event]().append),
        ):
            pass
        assert refused.value.code == Problem.BUSY


def test_a_port_that_is_taken_is_said_to_be(settings: Settings) -> None:
    other = dataclasses.replace(settings, home=settings.home.parent / "other")
    with running(settings) as server:
        taken = ServeHome(server.endpoint.port)
        with (
            pytest.raises(InvalidRequestError, match=f"port {taken.port} cannot be had") as refused,
            held(other, taken, "2.0.0", list[Event]().append),
        ):
            pass
        assert refused.value.field == "port"
