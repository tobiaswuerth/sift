import http.client
import json
import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import BinaryIO, cast

from sift.api.errors import StoreUnavailableError
from sift.api.events import Noted, Reporter
from sift.api.serving import Served
from sift.api.wire import Json
from sift.interfaces.mcp.protocol import MCP_PATH
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
STARTED = "started a home process for agents on port {port} (process {pid}); it leaves after {idle:,} s idle"
DETACHED = 0x00000008 | 0x00000200

type Start = Callable[[Settings, int], None]


def started(settings: Settings, idle: int) -> None:
    command = [sys.executable, "-m", "sift", "serve", "--port", "0", "--idle-exit", str(idle)]
    windows = sys.platform == "win32"
    _ = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=settings.home.parent if settings.home.parent.is_dir() else None,
        creationflags=DETACHED if windows else 0,
        start_new_session=not windows,
    )


@dataclass(frozen=True, slots=True)
class Starting:
    start: Start = started
    idle: int = IDLE_SECONDS
    patience: float = STARTING_SECONDS


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

    def _started(self) -> Endpoint:
        starting = self.starting
        starting.start(self.settings, starting.idle)
        looked = threading.Event()
        for _ in range(max(1, round(starting.patience / LOOK_EVERY_SECONDS))):
            endpoint = found(self.settings)
            if endpoint is not None:
                self.report(Noted(STARTED.format(port=endpoint.port, pid=endpoint.pid, idle=starting.idle)))
                return endpoint
            looked.wait(LOOK_EVERY_SECONDS)
        raise StoreUnavailableError(NOT_STARTED.format(seconds=starting.patience), LOG_REMEDY)


def _posted(endpoint: Endpoint, body: bytes) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection(LOCALHOST, endpoint.port, timeout=QUIET_SECONDS)
    try:
        connection.request("POST", MCP_PATH, body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def _asked_id(body: bytes) -> Json:
    try:
        read = cast("object", json.loads(body))
    except ValueError:
        return None
    return cast("Json", cast("dict[str, object]", read).get("id")) if isinstance(read, dict) else None


def _failed(body: bytes, message: str) -> bytes | None:
    if b'"id"' not in body or (at := _asked_id(body)) is None:
        return None
    return json.dumps({"jsonrpc": "2.0", "id": at, "error": {"code": INTERNAL, "message": message}}).encode()


def _relayed(home: _Home, body: bytes) -> bytes | None:
    try:
        try:
            status, answer = _posted(home.endpoint(), body)
        except (ConnectionError, TimeoutError, http.client.HTTPException):
            status, answer = _posted(home.endpoint(again=True), body)
    except StoreUnavailableError as error:
        return _failed(body, f"{error.message}; fix: {error.remedy}")
    except (ConnectionError, TimeoutError, http.client.HTTPException) as error:
        return _failed(body, SILENT.format(error=error))
    return None if status == http.client.ACCEPTED or not answer else answer


@dataclass(frozen=True, slots=True)
class _Answering:
    home: _Home
    pipes: Pipes
    lock: threading.Lock

    def __call__(self, body: bytes) -> None:
        answer = _relayed(self.home, body)
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
