from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from sift.api.events import Noted
from sift.interfaces.console.live import Live
from sift.settings import Settings

if TYPE_CHECKING:
    from sift.interfaces.cli.logs import HeardToo, Log
    from sift.interfaces.console.windows import Console
    from sift.interfaces.server.endpoint import Endpoint

LOGS = "sift.interfaces.cli.logs"
ENDPOINTS = "sift.interfaces.server.endpoint"
WINDOWS = "sift.interfaces.console.windows"


class Logs(Protocol):
    def logged(self, home: Path, command: str, now: datetime, heading: str) -> AbstractContextManager["Log"]: ...

    def heard_too(self, screen: Live, log: "Log", clock: Callable[[], float]) -> "HeardToo": ...


class Endpoints(Protocol):
    def found(self, settings: Settings, patience: float = ...) -> "Endpoint | None": ...

    def unshared(self, endpoint: "Endpoint", settings: Settings) -> tuple[Noted, ...]: ...


class Windows(Protocol):
    def console(self) -> "Console | None": ...

    def escapes_work(self, found: "Console | None", descriptor: int) -> bool: ...


def _as_logs(module: object) -> Logs:
    return cast("Logs", module)


def _as_endpoints(module: object) -> Endpoints:
    return cast("Endpoints", module)


def _as_windows(module: object) -> Windows:
    return cast("Windows", module)


def logs() -> Logs:
    return _as_logs(import_module(LOGS))


def endpoints() -> Endpoints:
    return _as_endpoints(import_module(ENDPOINTS))


def windows() -> Windows:
    return _as_windows(import_module(WINDOWS))
