import http.client
import json
import logging
import math
import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, cast

from sift.api.errors import Problem, StoreUnavailableError
from sift.api.events import Noted, Reporter
from sift.api.serving import Served
from sift.api.wire import Json
from sift.interfaces.mcp.protocol import MCP_PATH
from sift.interfaces.server.client import BUSY, BUSY_REMEDY, WholeRequests
from sift.interfaces.server.endpoint import Endpoint, found
from sift.settings import Settings

LOCALHOST = "127.0.0.1"
IDLE_SECONDS = 900
STARTING_SECONDS = 60.0
LOOK_EVERY_SECONDS = 0.05
ANSWERING = 8
QUIET_SECONDS = 600.0
INTERNAL = -32603
NOT_STARTED = "the home process did not start within {seconds:,} s"
LOG_REMEDY = "see its log in SIFT_HOME/logs"
SILENT = "the home process does not answer: {error}"
NO_JSON = "the home process answered {status} and no JSON"
NOT_PASSED_ON = "the message could not be passed to the home process"
JSON = "application/json"
STARTED = "started a home process for agents on port {port} (process {pid}); it leaves after {idle:,} s idle"
NEW_CONSOLE = 0x00000010 | 0x00000200
USE_SHOW_WINDOW = 0x00000001
MINIMIZED_UNFOCUSED = 7

type Start = Callable[[Settings, int], None]


def _shown_minimized() -> object | None:
    made = cast("Callable[..., object] | None", getattr(subprocess, "STARTUPINFO", None))
    return None if made is None else made(dwFlags=USE_SHOW_WINDOW, wShowWindow=MINIMIZED_UNFOCUSED)


def in_the_background(command: list[str], cwd: Path | None) -> subprocess.Popen[bytes]:
    windows = sys.platform == "win32"
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=cwd,
        creationflags=NEW_CONSOLE if windows else 0,
        startupinfo=_shown_minimized(),
        start_new_session=not windows,
    )


def started(settings: Settings, idle: int) -> None:
    command = [sys.executable, "-m", "sift", "serve", "--port", "0", "--idle-exit", str(idle)]
    _ = in_the_background(command, settings.home.parent if settings.home.parent.is_dir() else None)


@dataclass(frozen=True, slots=True)
class Starting:
    start: Start = started
    idle: int = IDLE_SECONDS
    patience: float = STARTING_SECONDS
    quiet: float = QUIET_SECONDS


STARTING = Starting()


@dataclass(frozen=True, slots=True)
class Pipes:
    read: BinaryIO
    write: BinaryIO


@dataclass(slots=True, eq=False)
class _Home:
    settings: Settings
    starting: Starting
    report: Reporter
    lock: threading.Lock = field(default_factory=threading.Lock)
    held: Endpoint | None = None

    def endpoint(self, *, again: bool = False) -> Endpoint:
        with self.lock:
            if self.held is None or again:
                self.held = found(self.settings) or self._started()
            return self.held

    def _listening(self) -> Endpoint | None:
        try:
            return found(self.settings, LOOK_EVERY_SECONDS)
        except StoreUnavailableError as error:
            if error.code is not Problem.BUSY:
                raise
            return None

    def _started(self) -> Endpoint:
        starting = self.starting
        starting.start(self.settings, starting.idle)
        looked = threading.Event()
        for _ in range(max(1, round(starting.patience / LOOK_EVERY_SECONDS))):
            endpoint = self._listening()
            if endpoint is not None:
                self.report(Noted(STARTED.format(port=endpoint.port, pid=endpoint.pid, idle=starting.idle)))
                return endpoint
            looked.wait(LOOK_EVERY_SECONDS)
        raise StoreUnavailableError(NOT_STARTED.format(seconds=starting.patience), LOG_REMEDY)


@dataclass(frozen=True, slots=True)
class _Posted:
    status: int
    json: bool
    body: bytes


def _posted(endpoint: Endpoint, body: bytes, quiet: float) -> _Posted:
    connection = WholeRequests(endpoint.port, quiet)
    try:
        connection.request("POST", MCP_PATH, body, {"Content-Type": JSON})
        response = connection.getresponse()
        kind = response.getheader("Content-Type") or ""
        return _Posted(response.status, kind.partition(";")[0].strip() == JSON, response.read())
    finally:
        connection.close()


def _finite(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(text[:40])
    return number


def _message(body: bytes) -> dict[str, object] | None:
    try:
        read = cast("object", json.loads(body, parse_float=_finite))
    except ValueError:
        return None
    return cast("dict[str, object]", read) if isinstance(read, dict) else None


def _asked_id(body: bytes) -> Json:
    return cast("Json", (_message(body) or {}).get("id"))


def _failed(body: bytes, message: str) -> bytes | None:
    at = _asked_id(body)
    if at is None:
        return None
    return json.dumps({"jsonrpc": "2.0", "id": at, "error": {"code": INTERNAL, "message": message}}).encode()


def _relayed(home: _Home, body: bytes) -> bytes | None:
    try:
        try:
            posted = _posted(home.endpoint(), body, home.starting.quiet)
        except (ConnectionError, http.client.HTTPException):
            posted = _posted(home.endpoint(again=True), body, home.starting.quiet)
    except StoreUnavailableError as error:
        return _failed(body, f"{error.message}; fix: {error.remedy}")
    except (ConnectionError, TimeoutError, http.client.HTTPException) as error:
        return _failed(body, SILENT.format(error=error))
    if posted.status == http.client.SERVICE_UNAVAILABLE:
        return _failed(body, f"{BUSY}; fix: {BUSY_REMEDY}")
    if posted.status == http.client.ACCEPTED or not posted.body:
        return None
    if not posted.json:
        return _failed(body, f"{NO_JSON.format(status=posted.status)}; fix: {LOG_REMEDY}")
    return _owned(body, posted.body)


def _owned(body: bytes, answer: bytes) -> bytes:
    at, read = _asked_id(body), _message(answer)
    if at is None or read is None or read.get("id") is not None:
        return answer
    return json.dumps({**read, "id": at}).encode()


def _answered(home: _Home, body: bytes) -> bytes | None:
    try:
        return _relayed(home, body)
    except RecursionError:
        raise
    except Exception as error:
        logging.getLogger(__name__).exception(NOT_PASSED_ON)
        return _failed(body, f"{NOT_PASSED_ON}: {type(error).__name__}: {error}")


@dataclass(frozen=True, slots=True)
class _Answering:
    home: _Home
    pipes: Pipes
    lock: threading.Lock

    def __call__(self, body: bytes) -> None:
        answer = _answered(self.home, body)
        if answer is not None:
            with self.lock:
                self.pipes.write.write(answer.rstrip(b"\r\n") + b"\n")
                self.pipes.write.flush()


def proxied(settings: Settings, report: Reporter, pipes: Pipes, starting: Starting = STARTING) -> Served:
    home = _Home(settings, starting, report)
    answering = _Answering(home, pipes, threading.Lock())
    with ThreadPoolExecutor(ANSWERING, thread_name_prefix="sift-stdio") as pool:
        for line in pipes.read:
            if line.strip():
                _ = pool.submit(answering, line.strip())
    held = home.held
    return Served("" if held is None else f"http://{LOCALHOST}:{held.port}")
