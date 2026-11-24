import contextlib
import logging
import os
import secrets
import socket
import threading
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import uvicorn

from sift.api.errors import InvalidRequestError, Problem, StoreUnavailableError
from sift.api.events import Listening, Reporter
from sift.api.serving import Served, ServeHome
from sift.app.kept_decoded import KeptDecoded
from sift.app.questions import RememberedQuestions
from sift.app.resident import ResidentCodes
from sift.interfaces.acts.acting import fresh_models
from sift.interfaces.server.app import ALIVE_SECONDS, LOGGER, Serving, served_app
from sift.interfaces.server.endpoint import ENDPOINT, SERVER, SERVING, Endpoint, found, written
from sift.interfaces.server.handled import HANDLED, Handled
from sift.interfaces.server.results import KeptResults
from sift.settings import Settings
from sift.storage.database import private_folder
from sift.storage.watching import HomeWatch
from sift.storage.writer import sole_writer

LOCALHOST = "127.0.0.1"
WORKERS = 32
ANOTHER_REMEDY = "commands are sent to it as they are; stop it to start another"
PORT_REMEDY = "give another port, or 0 for any free one"
IDLE_CHECK_SECONDS = 30.0
STOP_SECONDS = 5.0
RESTED = "the models were idle: they wait in RAM, off the GPU, until used again"


@dataclass(frozen=True, slots=True, eq=False)
class HomeServer:
    endpoint: Endpoint
    serving: Serving
    server: uvicorn.Server
    socket: socket.socket

    @property
    def url(self) -> str:
        return f"http://{LOCALHOST}:{self.endpoint.port}"


def _refuse_another(settings: Settings) -> None:
    another = found(settings.home)
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


@contextmanager
def held(
    settings: Settings,
    request: ServeHome,
    version: str,
    handled: Mapping[str, Handled] = HANDLED,
    alive_every: float = ALIVE_SECONDS,
) -> Generator[HomeServer]:
    private_folder(settings.home)
    _refuse_another(settings)
    models = fresh_models(settings)
    with (
        sole_writer(settings.home / SERVER, SERVING),
        _bound(request.port) as bound,
        ThreadPoolExecutor(WORKERS, thread_name_prefix="sift-serve") as pool,
        _resting(models.release_idle),
    ):
        (settings.home / ENDPOINT).unlink(missing_ok=True)
        port = cast("tuple[str, int]", bound.getsockname())[1]
        endpoint = Endpoint(port, os.getpid(), secrets.token_urlsafe(32), version)
        kept = (RememberedQuestions(models), ResidentCodes(), KeptDecoded(), KeptResults(), HomeWatch(settings.home))
        serving = Serving(settings, endpoint.token, *kept, threading.Lock(), pool, alive_every)
        config = uvicorn.Config(served_app(serving, handled), log_config=None, access_log=False, lifespan="off")
        written(settings.home, endpoint)
        try:
            yield HomeServer(endpoint, serving, uvicorn.Server(config), bound)
        finally:
            (settings.home / ENDPOINT).unlink(missing_ok=True)
            serving.watch.let_go()


def _served(running: HomeServer) -> None:
    running.server.run(sockets=[running.socket])


def serve(
    settings: Settings,
    request: ServeHome,
    version: str,
    report: Reporter,
    run: Callable[[HomeServer], None] = _served,
) -> Served:
    with held(settings, request, version) as running, contextlib.suppress(KeyboardInterrupt):
        report(Listening(running.url, str(settings.home)))
        run(running)
    return Served(running.url)
