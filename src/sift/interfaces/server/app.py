import hmac
import json
import logging
import queue
import threading
import time
from collections.abc import Awaitable, Callable, Generator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from sift.api.errors import InternalError, InvalidRequestError, NotFoundError, Problem, SiftError
from sift.api.events import Event, Noted, Reporter
from sift.api.ids import CollectionId
from sift.api.wire import Json, encoded, encoded_error
from sift.app.resident import ResidentCodes
from sift.interfaces.acts.acting import Acting
from sift.interfaces.server.handled import HANDLED, Handled
from sift.settings import Settings
from sift.storage.catalog import known_collections

if TYPE_CHECKING:
    from sift.inference.loading import Deferred

HOSTS = ("127.0.0.1", "localhost")
LINES = "application/x-ndjson; charset=utf-8"
ALIVE_SECONDS = 30.0
ALIVE = b'{"alive": true}\n'
WAITING = "another command writes to this home; this one waits for it to finish"
MOST_BODY = 1 << 20
LOGGER = "sift.server"
UNAUTHORIZED = InvalidRequestError(
    "no token, or a wrong one", "send the token in the home's server.json", code=Problem.UNAUTHORIZED
)
UNKNOWN = NotFoundError("no such command", "send one of sift's commands to /api/COMMAND")
NO_JSON = InvalidRequestError("the request is no JSON", "send the request as a JSON object")
TOO_LARGE = InvalidRequestError(f"the request is longer than {MOST_BODY:,} bytes", "send a shorter request")
FAILED_REMEDY = "see the home process's log in SIFT_HOME/logs, and report it"


@dataclass(frozen=True, slots=True, eq=False)
class Serving:
    settings: Settings
    token: str
    models: "Deferred"
    codes: ResidentCodes
    writing: threading.Lock
    pool: ThreadPoolExecutor
    alive_every: float = ALIVE_SECONDS


def _line(value: Json) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode() + b"\n"


def _refused(error: SiftError, status: int) -> JSONResponse:
    return JSONResponse({"error": encoded_error(error)}, status_code=status)


def _acting(serving: Serving, report: Callable[[Event], None]) -> Acting:
    return Acting(serving.settings, report, time.perf_counter, lambda: serving.models, serving.codes)


def _searched(home: Path) -> set[tuple[Path, CollectionId]]:
    return {(one.store, one.id) for one in known_collections(home) if one.active}


@contextmanager
def _turn(writing: threading.Lock, report: Reporter) -> Generator[None]:
    if not writing.acquire(blocking=False):
        report(Noted(WAITING))
        writing.acquire()
    try:
        yield
    finally:
        writing.release()


def _ran(serving: Serving, name: str, handled: Handled, data: object, lines: "queue.Queue[bytes | None]") -> None:
    def report(event: Event) -> None:
        lines.put(_line({"event": type(event).__name__, "data": encoded(event)}))

    started = time.perf_counter()
    try:
        with _turn(serving.writing, report) if handled.writes else nullcontext():
            try:
                answer = handled.run(_acting(serving, report), data)
            finally:
                if handled.writes:
                    serving.codes.kept_only(_searched(serving.settings.home))
        lines.put(_line({"answer": encoded(answer)}))
        logging.getLogger(LOGGER).info("%s answered in %.2f s", name, time.perf_counter() - started)
    except SiftError as error:
        lines.put(_line({"error": encoded_error(error)}))
        logging.getLogger(LOGGER).info("%s refused: %s", name, error.code.value)
    except RecursionError:
        raise
    except Exception as error:
        logging.getLogger(LOGGER).exception("%s failed", name)
        failed = InternalError(f"the home process failed: {type(error).__name__}: {error}", FAILED_REMEDY)
        lines.put(_line({"error": encoded_error(failed)}))
    finally:
        lines.put(None)


def _streamed(serving: Serving, name: str, handled: Handled, data: object) -> Iterator[bytes]:
    lines: queue.Queue[bytes | None] = queue.Queue()
    ran = serving.pool.submit(_ran, serving, name, handled, data, lines)
    while True:
        try:
            line = lines.get(timeout=serving.alive_every)
        except queue.Empty:
            yield ALIVE
            continue
        if line is None:
            break
        yield line
    ran.result()


async def _body(request: Request) -> bytes | None:
    read = bytearray()
    async for chunk in request.stream():
        read += chunk
        if len(read) > MOST_BODY:
            return None
    return bytes(read)


async def _command(serving: Serving, name: str, handled: Handled, request: Request) -> Response:
    given = (request.headers.get("authorization") or "").encode()
    if not hmac.compare_digest(given, f"Bearer {serving.token}".encode()):
        return _refused(UNAUTHORIZED, 401)
    body = await _body(request)
    if body is None:
        return _refused(TOO_LARGE, 413)
    try:
        data = cast("object", json.loads(body))
    except ValueError:
        return _refused(NO_JSON, 400)
    return StreamingResponse(_streamed(serving, name, handled, data), media_type=LINES)


async def _unknown(_request: Request, _error: Exception) -> Response:
    return _refused(UNKNOWN, 404)


def _route(serving: Serving, name: str, handled: Handled) -> Route:
    answer: Callable[[Request], Awaitable[Response]] = partial(_command, serving, name, handled)
    return Route(f"/api/{name}", answer, methods=["POST"])


def served_app(serving: Serving, handled: Mapping[str, Handled] = HANDLED) -> Starlette:
    return Starlette(
        routes=[_route(serving, name, each) for name, each in handled.items()],
        middleware=[Middleware(TrustedHostMiddleware, allowed_hosts=list(HOSTS))],
        exception_handlers={404: _unknown, 405: _unknown},
    )
