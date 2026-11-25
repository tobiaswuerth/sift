import json
import re
import socket
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import cast

import pytest

from sift.api.errors import InternalError, InvalidRequestError, NotFoundError, Problem, StoreUnavailableError
from sift.api.events import Advanced, Event, Noted
from sift.api.links import ListUnresolved, Unresolved
from sift.api.wire import encoded_error
from sift.interfaces.acts.links import UNRESOLVED
from sift.interfaces.server.client import UNFINISHED, UNREADABLE, ClientLimits, event_of, sent
from sift.interfaces.server.endpoint import Endpoint

LINES = b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n"
CHUNKED = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n20\r\n" + b'{"alive": tr'
LENGTH = re.compile(rb"Content-Length: (\d+)", re.IGNORECASE)


def _request(connection: socket.socket) -> bytes:
    read = b""
    while b"\r\n\r\n" not in read:
        read += connection.recv(65536)
    head, _, body = read.partition(b"\r\n\r\n")
    length = LENGTH.search(head)
    while length is not None and len(body) < int(length.group(1)):
        body += connection.recv(65536)
    return head + b"\r\n\r\n" + body


@contextmanager
def answering(reply: bytes | None, heard: list[bytes]) -> Generator[Endpoint]:
    listening = socket.create_server(("127.0.0.1", 0))
    done = threading.Event()

    def answer() -> None:
        with listening.accept()[0] as connection:
            heard.append(_request(connection))
            if reply is None:
                done.wait(30)
            else:
                connection.sendall(reply)

    thread = threading.Thread(target=answer, daemon=True)
    thread.start()
    try:
        yield Endpoint(cast("tuple[str, int]", listening.getsockname())[1], 1, "the-token", "2.0.0")
    finally:
        done.set()
        thread.join(30)
        listening.close()


def asked(reply: bytes | None, quiet: float = 30.0) -> tuple[Unresolved, list[Event], list[bytes]]:
    said, heard = list[Event](), list[bytes]()
    with answering(reply, heard) as endpoint:
        answer = sent(endpoint, UNRESOLVED, ListUnresolved(k=7), said.append, ClientLimits(quiet=quiet))
    return answer, said, heard


def refused(reply: bytes | None, quiet: float = 30.0) -> str:
    with pytest.raises((InternalError, StoreUnavailableError)) as caught:
        asked(reply, quiet)
    return caught.value.message


def _refusal(status: str, body: bytes) -> bytes:
    return f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body


def test_a_command_sent_reports_its_events_passes_over_other_lines_and_decodes_its_answer() -> None:
    noted = b'{"event": "Noted", "data": {"text": "a note", "remedy": null}}\n'
    reply = LINES + b'{"alive": true}\n' + noted + b'{"answer": {"targets": [], "unsearched": []}}\n'
    answer, said, (request,) = asked(reply)
    assert (answer, said) == (Unresolved(()), [Noted("a note")])
    head, _, body = request.partition(b"\r\n\r\n")
    assert head.startswith(b"POST /api/unresolved HTTP/1.1\r\n")
    assert b"\r\nAuthorization: Bearer the-token\r\n" in head
    assert json.loads(body) == {"collections": [], "k": 7}


def test_an_error_answered_or_refused_is_raised_as_the_home_process_said_it() -> None:
    error = encoded_error(NotFoundError("no such document", "name one that is"))
    with pytest.raises(NotFoundError, match="no such document"):
        asked(LINES + json.dumps({"error": error}).encode() + b"\n")
    unauthorized = encoded_error(InvalidRequestError("no token", "send it", code=Problem.UNAUTHORIZED))
    with pytest.raises(InvalidRequestError) as caught:
        asked(_refusal("401 Unauthorized", json.dumps({"error": unauthorized}).encode()))
    assert (caught.value.message, caught.value.code) == ("no token", Problem.UNAUTHORIZED)


def test_an_answer_this_sift_cannot_read_is_said_to_be() -> None:
    for reply in (
        LINES + b"no json\n",
        LINES + b"[1]\n",
        _refusal("500 Internal Server Error", b"Internal Server Error"),
        _refusal("400 Bad Request", b'{"detail": "a bad request"}'),
        LINES + b'{"answer": {"targets": 3}}\n',
        LINES + b'{"event": "Noted", "data": {"words": "a note"}}\n',
    ):
        assert refused(reply) == UNREADABLE, reply


def test_an_answer_ended_before_its_answer_line_is_said_to_be_unfinished() -> None:
    for reply in (LINES + b'{"alive": true}\n', CHUNKED, LINES + b'{"answer": {"targ'):
        assert refused(reply) == UNFINISHED, reply


def test_a_line_longer_than_the_client_reads_is_refused_without_reading_it_whole() -> None:
    said, heard = list[Event](), list[bytes]()
    with (
        answering(LINES + b'{"answer": "' + b"x" * 100 + b'"}\n', heard) as endpoint,
        pytest.raises(InternalError, match="sent a line longer than 50 bytes"),
    ):
        sent(endpoint, UNRESOLVED, ListUnresolved(k=7), said.append, ClientLimits(longest=50))


def test_a_home_process_that_closes_or_stays_silent_is_said_not_to_answer() -> None:
    assert refused(b"").endswith(" does not answer: Remote end closed connection without response")
    assert refused(None, quiet=0.05).endswith(" does not answer: timed out")


def test_an_event_of_a_newer_sift_is_passed_over_and_one_with_more_read() -> None:
    assert (event_of("JobQueued", {"number": 2}), event_of("Advanced", {"done": 3, "of": 9})) == (None, Advanced(3))
    newer = b'{"event": "JobQueued", "data": {"number": 2}}\n'
    answer, said, _ = asked(LINES + newer + b'{"answer": {"targets": []}}\n')
    assert (answer, said) == (Unresolved(()), [])
