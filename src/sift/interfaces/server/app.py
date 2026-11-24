import asyncio
import hmac
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from sift.api.errors import InternalError, InvalidRequestError, NotFoundError, Problem, SiftError, system_failure
from sift.api.events import Event, Noted, Reporter
from sift.api.ids import CollectionId
from sift.api.wire import Json, encoded, encoded_error
from sift.app.kept_decoded import KeptDecoded
from sift.app.resident import ResidentCodes
from sift.domain.text.json_values import deeper_than
from sift.domain.text.wording import counted
from sift.interfaces.acts.acting import Acting
from sift.interfaces.server.gate import Gate
from sift.interfaces.server.handled import HANDLED, Handled
from sift.interfaces.server.jobs import HeldJob, Jobs
from sift.interfaces.server.progress import Coalesced
from sift.interfaces.server.results import KeptResults, Moment
from sift.settings import Settings
from sift.storage.catalog import known_collections
from sift.storage.watching import HomeWatch

if TYPE_CHECKING:
    from sift.inference.models import Models

HOSTS = ("127.0.0.1", "localhost")
LINES = "application/x-ndjson; charset=utf-8"
ALIVE_SECONDS = 30.0
ALIVE = b'{"alive": true}\n'
END = b""
WAITING = "this command waits for {count} ahead of it: sift {command} (job {number}) runs now; see them with sift jobs"
MOST_BODY = 1 << 20
MOST_DEPTH = 64
NO_JSON_REMEDY = "send the request as a JSON object"
LOGGER = "sift.server"
UNAUTHORIZED = InvalidRequestError(
    "no token, or a wrong one", "send the token sift serve keeps in its user's own folder", code=Problem.UNAUTHORIZED
)
UNKNOWN = NotFoundError("no such command", "send one of sift's commands to /api/COMMAND")
NO_JSON = InvalidRequestError("the request is no JSON", NO_JSON_REMEDY)
TOO_LARGE = InvalidRequestError(f"the request is longer than {MOST_BODY:,} bytes", "send a shorter request")
FAILED_REMEDY = "see the home process's log in SIFT_HOME/logs, and report it"


@dataclass(frozen=True, slots=True, eq=False)
class Serving:
    settings: Settings
    token: str
    models: "Models"
    codes: ResidentCodes
    decoded: KeptDecoded
    results: KeptResults
    watch: HomeWatch
    writer: ThreadPoolExecutor
    pool: ThreadPoolExecutor
    alive_every: float = ALIVE_SECONDS
    gate: Gate = field(default_factory=Gate)
    jobs: Jobs = field(default_factory=Jobs)
    clock: Callable[[], float] = time.perf_counter


def _line(value: Json) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode() + b"\n"


def _no_number(name: str) -> object:
    raise InvalidRequestError(f"the request holds {name}, which is no JSON number", NO_JSON_REMEDY)


def _given_once(pairs: list[tuple[str, object]]) -> dict[str, object]:
    held = dict[str, object]()
    for key, value in pairs:
        if key in held:
            raise InvalidRequestError(f"the request gives the key {key[:40]!r} twice", NO_JSON_REMEDY)
        held[key] = value
    return held


def json_of(body: bytes) -> object:
    text = body.decode()
    if deeper_than(text, MOST_DEPTH):
        raise InvalidRequestError(f"the request is nested more than {MOST_DEPTH} deep", NO_JSON_REMEDY)
    return cast("object", json.loads(text, parse_constant=_no_number, object_pairs_hook=_given_once))


def _refused(error: SiftError, status: int) -> JSONResponse:
    return JSONResponse({"error": encoded_error(error)}, status_code=status)


def _acting(serving: Serving, report: Reporter) -> Acting:
    return Acting(
        serving.settings,
        report,
        time.perf_counter,
        lambda: serving.models,
        serving.codes,
        serving.decoded,
        serving.jobs,
    )


def _searched(home: Path) -> set[tuple[Path, CollectionId]]:
    return {(one.store, one.id) for one in known_collections(home) if one.active}


def _written(serving: Serving, handled: Handled, data: object, report: Reporter) -> object:
    with serving.gate.alone(report) if handled.alone else nullcontext():
        serving.results.written()
        serving.watch.let_go()
        try:
            return handled.run(_acting(serving, report), data)
        finally:
            searched = _searched(serving.settings.home)
            serving.codes.kept_only(searched)
            serving.decoded.kept_only(searched)
            serving.results.written()


def _moment(serving: Serving) -> Moment:
    return Moment(serving.results.seen(), serving.watch.seen())


def _read(serving: Serving, handled: Handled, data: object, report: Reporter, noted: threading.Event) -> object:
    with serving.gate.shared(report):
        return _answered(serving, handled, data, report, noted)


def _answered(serving: Serving, handled: Handled, data: object, report: Reporter, noted: threading.Event) -> object:
    if handled.kept is None:
        return handled.run(_acting(serving, report), data)
    key = handled.kept(data)
    before = _moment(serving)
    found = serving.results.found(key, before)
    if found is not None:
        return found
    answer = handled.run(_acting(serving, report), data)
    if not noted.is_set():
        serving.results.keep(key, answer, before, _moment(serving))
    return answer


@dataclass(frozen=True, slots=True)
class _Sent:
    name: str
    handled: Handled
    data: object


def _answer(serving: Serving, asked: _Sent, report: Reporter, noted: threading.Event, job: HeldJob | None) -> object:
    if job is None:
        return _read(serving, asked.handled, asked.data, report, noted)
    serving.jobs.began(job)
    return _written(serving, asked.handled, asked.data, serving.jobs.reporting(job, report))


@dataclass(frozen=True, slots=True)
class _Lines:
    loop: asyncio.AbstractEventLoop
    queue: "asyncio.Queue[tuple[bytes, ...]]"
    gone: threading.Event

    def put(self, *lines: bytes) -> None:
        if not self.gone.is_set() and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.queue.put_nowait, lines)


def _final(serving: Serving, sent: _Sent, report: Reporter, noted: threading.Event, job: HeldJob | None) -> bytes:
    started = time.perf_counter()
    try:
        answer = _answer(serving, sent, report, noted, job)
    except (SiftError, OSError) as error:
        refused = system_failure(error) if isinstance(error, OSError) else error
        logging.getLogger(LOGGER).info("%s refused: %s", sent.name, refused.code.value)
        return _line({"error": encoded_error(refused)})
    except RecursionError:
        raise
    except Exception as error:
        logging.getLogger(LOGGER).exception("%s failed", sent.name)
        failed = InternalError(f"the home process failed: {type(error).__name__}: {error}", FAILED_REMEDY)
        return _line({"error": encoded_error(failed)})
    logging.getLogger(LOGGER).info("%s answered in %.2f s", sent.name, time.perf_counter() - started)
    return _line({"answer": encoded(answer)})


def _ran(serving: Serving, sent: _Sent, lines: _Lines, job: HeldJob | None) -> None:
    noted = threading.Event()

    def said(event: Event) -> None:
        if isinstance(event, Noted):
            noted.set()
        lines.put(_line({"event": type(event).__name__, "data": encoded(event)}))

    report, final = Coalesced(said, serving.clock), tuple[bytes, ...]()
    try:
        final = (_final(serving, sent, report, noted, job),)
    finally:
        report.flush()
        if job is not None:
            serving.jobs.ended(job)
        lines.put(*final, END)


def _waiting(serving: Serving, job: HeldJob) -> Iterator[bytes]:
    if ahead := serving.jobs.ahead(job):
        said = WAITING.format(count=counted(len(ahead), "job"), command=ahead[0].command, number=ahead[0].number)
        yield _line({"event": Noted.__name__, "data": encoded(Noted(said))})


async def _streamed(serving: Serving, name: str, handled: Handled, data: object) -> AsyncIterator[bytes]:
    lines = _Lines(asyncio.get_running_loop(), asyncio.Queue(), threading.Event())
    job = serving.jobs.queued(name) if handled.writes else None
    if job is not None:
        for line in _waiting(serving, job):
            yield line
    ran = (serving.pool if job is None else serving.writer).submit(
        _ran, serving, _Sent(name, handled, data), lines, job
    )
    getting = asyncio.ensure_future(lines.queue.get())
    try:
        while True:
            if not (await asyncio.wait({getting}, timeout=serving.alive_every))[0]:
                yield ALIVE
                continue
            batch = getting.result()
            for line in batch:
                if line != END:
                    yield line
            if END in batch:
                break
            getting = asyncio.ensure_future(lines.queue.get())
    finally:
        lines.gone.set()
        getting.cancel()
        if job is not None and not ran.done():
            job.watched = False
    await asyncio.wrap_future(ran)


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
        data = json_of(body)
    except InvalidRequestError as refused:
        return _refused(refused, 400)
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
