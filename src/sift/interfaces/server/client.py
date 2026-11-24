import http.client
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from sift.api.errors import InternalError, InvalidRequestError, SiftError, StoreUnavailableError
from sift.api.events import EVENT_KINDS, Event, Reporter
from sift.api.wire import Json, decoded, decoded_error, encoded
from sift.interfaces.acts.acting import Act, answer_kind
from sift.interfaces.server.endpoint import Endpoint

LOCALHOST = "127.0.0.1"
QUIET_SECONDS = 120.0
MOST_LINE = 1 << 30
UNFINISHED = "the home process ended its answer unfinished"
UNREADABLE = "the home process answered what this sift cannot read"
LOG_REMEDY = "see its log in SIFT_HOME/logs, and ask again"
SILENT_REMEDY = "see whether sift serve still runs there; else stop it and start it again"


@dataclass(frozen=True, slots=True)
class ClientLimits:
    quiet: float = QUIET_SECONDS
    longest: int = MOST_LINE


NO_LONGER = ClientLimits()


def event_of(name: str, data: Json) -> Event:
    return decoded(EVENT_KINDS[name], data)


def _readable[**P, T](decode: Callable[P, T], *given: P.args, **named: P.kwargs) -> T:
    try:
        return decode(*given, **named)
    except (InvalidRequestError, KeyError, ValueError, TypeError):
        raise InternalError(UNREADABLE, LOG_REMEDY) from None


def _said(line: bytes) -> dict[str, Json]:
    try:
        read = cast("object", json.loads(line))
    except ValueError:
        read = None
    if not isinstance(read, dict):
        raise InternalError(UNREADABLE, LOG_REMEDY)
    return cast("dict[str, Json]", read)


def _refusal(response: http.client.HTTPResponse, longest: int) -> SiftError:
    said = _said(response.read(longest))
    return _readable(decoded_error, said["error"]) if "error" in said else InternalError(UNREADABLE, LOG_REMEDY)


def _line(response: http.client.HTTPResponse, longest: int) -> bytes:
    line = response.readline(longest + 1)
    if line and not line.endswith(b"\n"):
        said = f"the home process sent a line longer than {longest:,} bytes" if len(line) > longest else UNFINISHED
        raise InternalError(said, LOG_REMEDY)
    return line


def _answer[R, A](act: Act[R, A], response: http.client.HTTPResponse, report: Reporter, longest: int) -> A:
    while line := _line(response, longest):
        said = _said(line)
        if "event" in said:
            report(_readable(event_of, str(said["event"]), said.get("data")))
        elif "answer" in said:
            return _readable(decoded, answer_kind(act), said["answer"])
        elif "error" in said:
            raise _readable(decoded_error, said["error"])
    raise InternalError(UNFINISHED, LOG_REMEDY)


def sent[R, A](endpoint: Endpoint, act: Act[R, A], request: R, report: Reporter, limits: ClientLimits = NO_LONGER) -> A:
    connection = http.client.HTTPConnection(LOCALHOST, endpoint.port, timeout=limits.quiet)
    headers = {"Authorization": f"Bearer {endpoint.token}", "Content-Type": "application/json"}
    try:
        connection.request("POST", f"/api/{act.name}", json.dumps(encoded(request), allow_nan=False).encode(), headers)
        with connection.getresponse() as response:
            if response.status != http.client.OK:
                raise _refusal(response, limits.longest)
            return _answer(act, response, report, limits.longest)
    except http.client.IncompleteRead:
        raise InternalError(UNFINISHED, LOG_REMEDY) from None
    except (ConnectionError, TimeoutError) as error:
        said = f"the home process on port {endpoint.port} does not answer: {error}"
        raise StoreUnavailableError(said, SILENT_REMEDY) from None
    finally:
        connection.close()
