from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast, get_type_hints

from sift.api.errors import NotFoundError
from sift.api.events import Reporter
from sift.api.jobs import ServedJob
from sift.app.resident import CodeSource, DecodedSource, decoded_each_search, read_each_time
from sift.settings import Settings

if TYPE_CHECKING:
    from sift.inference.loading import Deferred
    from sift.inference.models import Models

LOADING = "sift.inference.loading"


NO_HOME_PROCESS = "no home process runs, so no job runs in the background"
SERVE_REMEDY = "jobs run in the background of sift serve alone"


class JobBoard(Protocol):
    def listed(self) -> tuple[ServedJob, ...]: ...

    def stopped(self, number: int) -> ServedJob: ...


@dataclass(frozen=True, slots=True)
class NoJobs:
    def listed(self) -> tuple[ServedJob, ...]:
        return ()

    def stopped(self, number: int) -> ServedJob:
        raise NotFoundError(f"there is no job {number}: {NO_HOME_PROCESS}", SERVE_REMEDY)


NO_JOBS = NoJobs()


@dataclass(frozen=True, slots=True)
class Acting:
    settings: Settings
    report: Reporter
    clock: Callable[[], float]
    models: "Callable[[], Models]"
    codes: CodeSource = read_each_time
    decoded: DecodedSource = decoded_each_search
    jobs: JobBoard = NO_JOBS


@dataclass(frozen=True, slots=True)
class Act[R, A]:
    name: str
    request: type[R]
    run: Callable[[Acting, R], A]
    writes: bool = False


def answer_kind[R, A](act: Act[R, A]) -> type[A]:
    return cast("type[A]", cast("dict[str, object]", get_type_hints(act.run))["return"])


class Loading(Protocol):
    def deferred(self, model: str, reranker: str, gpus: str | None) -> "Deferred": ...


def _as_loading(module: object) -> Loading:
    return cast("Loading", module)


def fresh_models(settings: Settings) -> "Deferred":
    return _as_loading(import_module(LOADING)).deferred(settings.embed_model, settings.rerank_model, settings.gpus)
