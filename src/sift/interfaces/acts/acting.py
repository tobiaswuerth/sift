from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast, get_type_hints

from sift.api.events import Reporter
from sift.app.resident import CodeSource, DecodedSource, decoded_each_search, read_each_time
from sift.settings import Settings

if TYPE_CHECKING:
    from sift.inference.loading import Deferred
    from sift.inference.models import Models

LOADING = "sift.inference.loading"


@dataclass(frozen=True, slots=True)
class Acting:
    settings: Settings
    report: Reporter
    clock: Callable[[], float]
    models: "Callable[[], Models]"
    codes: CodeSource = read_each_time
    decoded: DecodedSource = decoded_each_search


@dataclass(frozen=True, slots=True)
class Act[R, A]:
    name: str
    request: type[R]
    run: Callable[[Acting, R], A]


def answer_kind[R, A](act: Act[R, A]) -> type[A]:
    return cast("type[A]", cast("dict[str, object]", get_type_hints(act.run))["return"])


class Loading(Protocol):
    def deferred(self, model: str, reranker: str, gpus: str | None) -> "Deferred": ...


def _as_loading(module: object) -> Loading:
    return cast("Loading", module)


def fresh_models(settings: Settings) -> "Deferred":
    return _as_loading(import_module(LOADING)).deferred(settings.embed_model, settings.rerank_model, settings.gpus)
