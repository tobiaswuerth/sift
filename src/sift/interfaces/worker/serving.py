import socket
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import uvicorn

from sift.api.events import Lending, Reporter
from sift.api.serving import ALL_INTERFACES, Served, ServeWorker
from sift.app.loading import prepared
from sift.domain.codec.codes import width_of
from sift.domain.text.embedding import RECIPE
from sift.inference.loading import Resting
from sift.interfaces.acts.acting import own_models
from sift.interfaces.server.heads import HeadsInTime
from sift.interfaces.server.serving import (
    IDLE_CHECK_SECONDS,
    bound,
    on_a_selector,
    resting,
    stopped_as_by_ctrl_c,
)
from sift.interfaces.worker.app import Lent, worker_app
from sift.interfaces.worker.keys import FOLDER, WorkerKey, worker_key
from sift.settings import Settings
from sift.storage.database import private_folder
from sift.storage.writer import sole_writer

WORKING = "sift worker"
WIDTH_PROBE = "the width of a vector"
GRACE_SECONDS = 10
KEEP_ALIVE_SECONDS = 5
MOST_CONNECTIONS = 64


@dataclass(frozen=True, slots=True, eq=False)
class WorkerServer:
    key: WorkerKey
    server: uvicorn.Server
    socket: socket.socket
    port: int


def worker_served_until_stopped(running: WorkerServer) -> None:
    on_a_selector(running.server.serve(sockets=[running.socket]))


@dataclass(frozen=True, slots=True)
class Lend:
    run: Callable[[WorkerServer], None] = worker_served_until_stopped
    models: Callable[[Settings], Resting] = own_models
    machine: Callable[[], str] = socket.gethostname
    bind: Callable[[int, str], socket.socket] = bound
    grace_seconds: int = GRACE_SECONDS
    keep_alive_seconds: int = KEEP_ALIVE_SECONDS
    idle_every: float = IDLE_CHECK_SECONDS


LEND = Lend()


def _said(settings: Settings, models: Resting, version: str) -> dict[str, object]:
    width = width_of(models.encode([WIDTH_PROBE], query=False))
    return {
        "model": models.name,
        "width": width,
        "reranker": settings.rerank_model,
        "version": version,
        "recipe": RECIPE,
    }


@contextmanager
def lent(
    settings: Settings, request: ServeWorker, version: str, report: Reporter, how: Lend = LEND
) -> Generator[WorkerServer]:
    private_folder(settings.home)
    with sole_writer(settings.home / FOLDER, WORKING):
        key = worker_key(settings.home)
        models = how.models(settings)
        prepared(report, (models.prepare, models.prepare_ranker))
        said = _said(settings, models, version)
        with (
            how.bind(request.port, request.host) as listening,
            ThreadPoolExecutor(1, thread_name_prefix="sift-worker") as work,
            resting(models.release_idle, how.idle_every),
        ):
            config = uvicorn.Config(
                worker_app(Lent(key.token, models, said, work)),
                http=HeadsInTime,
                ssl_certfile=str(key.certificate),
                ssl_keyfile=str(key.private_key),
                log_config=None,
                access_log=False,
                lifespan="off",
                timeout_graceful_shutdown=how.grace_seconds,
                timeout_keep_alive=how.keep_alive_seconds,
                limit_concurrency=MOST_CONNECTIONS,
            )
            port = cast("tuple[str, int]", listening.getsockname())[1]
            yield WorkerServer(key, uvicorn.Server(config), listening, port)


def lend(settings: Settings, request: ServeWorker, version: str, report: Reporter, how: Lend = LEND) -> Served:
    machine = how.machine() if request.host == ALL_INTERFACES else request.host
    with stopped_as_by_ctrl_c(), lent(settings, request, version, report, how) as running:
        try:
            report(Lending(machine, running.port, running.key.key))
            how.run(running)
        except KeyboardInterrupt:
            running.server.should_exit = True
    return Served(f"https://{machine}:{running.port}")
