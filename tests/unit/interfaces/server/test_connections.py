import http.client
import select
import socket

import pytest

from sift.api.errors import Problem, StoreUnavailableError
from sift.api.links import ListUnresolved
from sift.interfaces.acts.links import UNRESOLVED
from sift.interfaces.server.client import BUSY, BUSY_REMEDY, sent
from sift.interfaces.server.serving import HomeServer, Serve
from sift.settings import Settings
from tests.serving import running

HEAD = b"POST /api/status HTTP/1.1\r\nHost: 127.0.0.1\r\n"
TRICKLED = HEAD + b"X-Slow: " + b"a" * 400
QUICKLY = Serve(keep_alive_seconds=1)
LONGEST_WAIT = 10.0


def opened(server: HomeServer) -> socket.socket:
    return socket.create_connection(("127.0.0.1", server.endpoint.port), timeout=LONGEST_WAIT)


def closed(left: socket.socket) -> bool:
    try:
        return left.recv(1) == b""
    except (ConnectionResetError, ConnectionAbortedError):
        return True
    except TimeoutError:
        return False


def kept(server: HomeServer) -> http.client.HTTPConnection:
    connection = http.client.HTTPConnection("127.0.0.1", server.endpoint.port, timeout=LONGEST_WAIT)
    connection.request("POST", "/api/status", b"{}", {"Authorization": f"Bearer {server.endpoint.token}"})
    with connection.getresponse() as response:
        assert (response.status, response.read().endswith(b"\n")) == (http.client.OK, True)
    return connection


def test_a_request_head_unfinished_for_long_is_closed(settings: Settings) -> None:
    with running(settings, how=QUICKLY) as server, opened(server) as left:
        left.sendall(HEAD)
        assert closed(left)


def test_a_head_trickled_a_byte_at_a_time_is_closed_by_its_deadline_all_the_same(settings: Settings) -> None:
    with running(settings, how=QUICKLY) as server, opened(server) as left:
        ended = None
        for at in range(len(TRICKLED)):
            if select.select([left], [], [], 0.05)[0]:
                ended = at
                break
            left.sendall(TRICKLED[at : at + 1])
        assert (ended is not None, closed(left)) == (True, True)


def test_a_second_head_unfinished_for_long_on_a_kept_connection_is_closed(settings: Settings) -> None:
    with running(settings, how=QUICKLY) as server:
        connection = kept(server)
        try:
            left = connection.sock
            left.sendall(HEAD)
            assert closed(left)
        finally:
            connection.close()


def test_past_its_most_connections_the_home_process_is_busy_and_says_so(settings: Settings) -> None:
    with running(settings, how=Serve(most_connections=3)) as server:
        held = [kept(server), kept(server)]
        try:
            with pytest.raises(StoreUnavailableError) as refused:
                sent(server.endpoint, UNRESOLVED, ListUnresolved(), lambda _: None)
        finally:
            for connection in held:
                connection.close()
        assert (str(refused.value), refused.value.code, refused.value.remedy) == (BUSY, Problem.BUSY, BUSY_REMEDY)
