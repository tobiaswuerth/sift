import logging
import os
import secrets
import signal
import socket
import threading
import time
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import FrameType
from typing import cast, override

import uvicorn

from sift.api.errors import InvalidRequestError, Problem, StoreUnavailableError
from sift.api.events import Listening, Noted, Reporter
from sift.api.serving import Served, ServeHome
from sift.api.wire import CONTRACT
from sift.app.kept_decoded import KeptDecoded
from sift.app.questions import RememberedQuestions
from sift.app.resident import ResidentCodes
from sift.interfaces.acts.acting import fresh_models
from sift.interfaces.server.app import ALIVE_SECONDS, LOGGER, Serving, served_app
from sift.interfaces.server.endpoint import SERVER, SERVING, Endpoint, found, own_endpoint, written
from sift.interfaces.server.handled import HANDLED, Handled
from sift.interfaces.server.heads import HeadsInTime
from sift.interfaces.server.results import KeptResults
from sift.settings import Settings, acting_on
from sift.storage.database import private_folder
from sift.storage.watching import HomeWatch
from sift.storage.writer import sole_writer

LOCALHOST = "127.0.0.1"
WORKERS = 32
ANOTHER_REMEDY = "commands are sent to it as they are; stop it to start another"
PORT_REMEDY = "give another port, or 0 for any free one"
IDLE_CHECK_SECONDS = 30.0
STOP_SECONDS = 5.0
GRACE_SECONDS = 10
KEEP_ALIVE_SECONDS = 5
MOST_CONNECTIONS = 256
STOPPING = "stopping: {jobs} stops at its next step; what it wrote so far is kept"
RESTED = "the models were idle: they wait in RAM, off the GPU, until used again"
STOPPING_SIGNALS: tuple[int, ...] = tuple(
    cast("int", getattr(signal, name)) for name in ("SIGTERM", "SIGBREAK") if hasattr(signal, name)
)


def _stopping(serving: Serving, listed: Path, report: Reporter) -> None:
    listed.unlink(missing_ok=True)
    if stopped := serving.jobs.stop_all():
        jobs = ", ".join(f"sift {job.command} (job {job.number})" for job in stopped)
        report(Noted(STOPPING.format(jobs=jobs)))


class StoppingServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, on_first_exit: Callable[[], None]) -> None:
        super().__init__(config)
        self.on_first_exit: Callable[[], None] = on_first_exit

    @override
    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        if not self.should_exit:
            self.on_first_exit()
        super().handle_exit(sig, frame)


@dataclass(frozen=True, slots=True, eq=False)
class HomeServer:
    endpoint: Endpoint
    serving: Serving
    server: StoppingServer
    socket: socket.socket
    listed: Path

    @property
    def url(self) -> str:
        return f"http://{LOCALHOST}:{self.endpoint.port}"


def _refuse_another(settings: Settings) -> None:
    another = found(settings)
    if another is not None:
        said = f"another sift serves this home on port {another.port} (process {another.pid})"
        raise StoreUnavailableError(said, ANOTHER_REMEDY, code=Problem.BUSY)


def _bound(port: int) -> socket.socket:
    try:
        return socket.create_server((LOCALHOST, port))
    except OSError as error:
        raise InvalidRequestError(
            f"port {port} cannot be had: {error.strerror or error}", PORT_REMEDY, "port"
        ) from None


def watched_idle(release: Callable[[], bool], stop: threading.Event, every: float = IDLE_CHECK_SECONDS) -> None:
    while not stop.wait(every):
        if release():
            logging.getLogger(LOGGER).info(RESTED)


@contextmanager
def _resting(release: Callable[[], bool]) -> Generator[None]:
    stop = threading.Event()
    watch = threading.Thread(target=watched_idle, args=(release, stop), name="sift-idle", daemon=True)
    watch.start()
    try:
        yield
    finally:
        stop.set()
        watch.join(STOP_SECONDS)


def _served(running: HomeServer) -> None:
    running.server.run(sockets=[running.socket])


@dataclass(frozen=True, slots=True)
class Serve:
    run: Callable[[HomeServer], None] = _served
    handled: Mapping[str, Handled] = HANDLED
    alive_every: float = ALIVE_SECONDS
    grace_seconds: int = GRACE_SECONDS
    clock: Callable[[], float] = time.perf_counter
    keep_alive_seconds: int = KEEP_ALIVE_SECONDS
    most_connections: int = MOST_CONNECTIONS


SERVE = Serve()


@contextmanager
def held(
    settings: Settings, request: ServeHome, version: str, report: Reporter, how: Serve = SERVE
) -> Generator[HomeServer]:
    private_folder(settings.home)
    listed = own_endpoint(settings)
    _refuse_another(settings)
    models = fresh_models(settings)
    with (
        sole_writer(settings.home / SERVER, SERVING),
        _bound(request.port) as bound,
        ThreadPoolExecutor(WORKERS, thread_name_prefix="sift-serve") as pool,
        ThreadPoolExecutor(1, thread_name_prefix="sift-write") as writer,
        _resting(models.release_idle),
    ):
        listed.unlink(missing_ok=True)
        port = cast("tuple[str, int]", bound.getsockname())[1]
        endpoint = Endpoint(port, os.getpid(), secrets.token_urlsafe(32), version, acting_on(settings), CONTRACT)
        kept = (RememberedQuestions(models), ResidentCodes(), KeptDecoded(), KeptResults(), HomeWatch(settings.home))
        serving = Serving(settings, endpoint.token, *kept, writer, pool, how.alive_every, clock=how.clock)
        app = served_app(serving, how.handled)
        config = uvicorn.Config(
            app,
            http=HeadsInTime,
            log_config=None,
            access_log=False,
            lifespan="off",
            timeout_graceful_shutdown=how.grace_seconds,
            timeout_keep_alive=how.keep_alive_seconds,
            limit_concurrency=how.most_connections,
        )
        written(listed, endpoint)
        server = StoppingServer(config, partial(_stopping, serving, listed, report))
        try:
            yield HomeServer(endpoint, serving, server, bound, listed)
        finally:
            listed.unlink(missing_ok=True)
            serving.watch.let_go()


def _interrupted(_signal: int, _frame: FrameType | None) -> None:
    raise KeyboardInterrupt


@contextmanager
def _stopped_as_by_ctrl_c() -> Generator[None]:
    kept = {each: signal.signal(each, _interrupted) for each in STOPPING_SIGNALS}
    try:
        yield
    finally:
        for each, handler in kept.items():
            signal.signal(each, handler)


def serve(settings: Settings, request: ServeHome, version: str, report: Reporter, how: Serve = SERVE) -> Served:
    with _stopped_as_by_ctrl_c(), held(settings, request, version, report, how) as running:
        try:
            report(Listening(running.url, str(settings.home)))
            how.run(running)
        except KeyboardInterrupt:
            running.server.should_exit = True
            _stopping(running.serving, running.listed, report)
    return Served(running.url)
