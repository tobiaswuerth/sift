from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from sift.api.events import Noted
from sift.interfaces.console.live import Live
from sift.late import late
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


def logs() -> Logs:
    return late(LOGS, Logs)


def endpoints() -> Endpoints:
    return late(ENDPOINTS, Endpoints)


def windows() -> Windows:
    return late(WINDOWS, Windows)


if TYPE_CHECKING:
    from sift.interfaces.cli import logs as _logs
    from sift.interfaces.console import windows as _windows
    from sift.interfaces.server import endpoint as _endpoint

    _LOGS: Logs = _logs
    _ENDPOINTS: Endpoints = _endpoint
    _WINDOWS: Windows = _windows
