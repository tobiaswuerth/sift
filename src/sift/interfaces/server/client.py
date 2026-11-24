import http.client
import json
from typing import cast

from sift.api.errors import InternalError, SiftError, StoreUnavailableError
from sift.api.events import EVENT_KINDS, Event, Reporter
from sift.api.wire import Json, decoded, decoded_error, encoded
from sift.interfaces.acts.acting import Act, answer_kind
from sift.interfaces.server.endpoint import Endpoint

LOCALHOST = "127.0.0.1"
QUIET_SECONDS = 120.0
UNFINISHED = "the home process ended its answer unfinished"
UNREADABLE = "the home process answered what this sift cannot read"
LOG_REMEDY = "see its log in SIFT_HOME/logs, and ask again"
SILENT_REMEDY = "see whether sift serve still runs there; else stop it and start it again"


def event_of(name: str, data: Json) -> Event:
    return decoded(EVENT_KINDS[name], data)


def _said(line: bytes) -> dict[str, Json]:
    try:
        read = cast("object", json.loads(line))
    except ValueError:
        read = None
    if not isinstance(read, dict):
        raise InternalError(UNREADABLE, LOG_REMEDY)
    return cast("dict[str, Json]", read)


def _refusal(response: http.client.HTTPResponse) -> SiftError:
    said = _said(response.read())
    return decoded_error(said["error"]) if "error" in said else InternalError(UNREADABLE, LOG_REMEDY)


def _answer[R, A](act: Act[R, A], response: http.client.HTTPResponse, report: Reporter) -> A:
    while line := response.readline():
        said = _said(line)
        if "event" in said:
            report(event_of(str(said["event"]), said["data"]))
        elif "answer" in said:
            return decoded(answer_kind(act), said["answer"])
        elif "error" in said:
            raise decoded_error(said["error"])
    raise InternalError(UNFINISHED, LOG_REMEDY)


def sent[R, A](endpoint: Endpoint, act: Act[R, A], request: R, report: Reporter, quiet: float = QUIET_SECONDS) -> A:
    connection = http.client.HTTPConnection(LOCALHOST, endpoint.port, timeout=quiet)
    headers = {"Authorization": f"Bearer {endpoint.token}", "Content-Type": "application/json"}
    try:
        connection.request("POST", f"/api/{act.name}", json.dumps(encoded(request)).encode(), headers)
        with connection.getresponse() as response:
            if response.status != http.client.OK:
                raise _refusal(response)
            return _answer(act, response, report)
    except http.client.IncompleteRead:
        raise InternalError(UNFINISHED, LOG_REMEDY) from None
    except (ConnectionError, TimeoutError) as error:
        said = f"the home process on port {endpoint.port} does not answer: {error}"
        raise StoreUnavailableError(said, SILENT_REMEDY) from None
    finally:
        connection.close()
