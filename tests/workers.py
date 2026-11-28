import http.client
import http.server
import json
import socket
import ssl
import threading
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import cast, final

from sift.api.events import Event
from sift.api.serving import ServeWorker, WorkerAddress
from sift.inference.loading import Resting
from sift.inference.remote import Remote, Resolve, addresses, unverified_tls
from sift.interfaces.worker.keys import WorkerKey
from sift.interfaces.worker.serving import LEND, Lend, WorkerServer, lent, worker_served_until_stopped
from sift.settings import Settings
from tests.serving import VERSION
from tests.waits import joined, until

LOOPBACK = "127.0.0.1"
QUICK = replace(LEND, grace_seconds=1)


@dataclass(frozen=True, slots=True)
class LiveWorker:
    running: WorkerServer
    heard: list[Event]
    reached: list[Remote] = field(default_factory=list[Remote])

    @property
    def address(self) -> WorkerAddress:
        return WorkerAddress(LOOPBACK, self.running.port, self.running.key.token, self.running.key.pin)

    def remote(
        self, release: str = VERSION, address: WorkerAddress | None = None, resolve: Resolve = addresses
    ) -> Remote:
        made = Remote(address or self.address, release, resolve)
        self.reached.append(made)
        return made

    def requests(self) -> int:
        return self.running.server.server_state.total_requests

    def connections(self) -> set[int]:
        return {id(each) for each in self.running.server.server_state.connections}

    def closing_all(self) -> bool:
        return all(each.transport.is_closing() for each in self.running.server.server_state.connections)


def offered(models: Resting) -> Callable[[Settings], Resting]:
    def given(_settings: Settings) -> Resting:
        return models

    return given


@contextmanager
def working(settings: Settings, models: Resting, version: str = VERSION, how: Lend = QUICK) -> Generator[LiveWorker]:
    heard = list[Event]()
    asked = ServeWorker(port=0, host=LOOPBACK)
    with lent(settings, asked, version, heard.append, replace(how, models=offered(models))) as running:
        thread = threading.Thread(target=worker_served_until_stopped, args=(running,), daemon=True)
        thread.start()
        assert until(lambda: running.server.started), "the worker did not start within 30 s"
        live = LiveWorker(running, heard)
        try:
            yield live
        finally:
            for each in live.reached:
                each.close()
            running.server.should_exit = True
            joined(thread, 30)


def asked(
    worker: LiveWorker, method: str, path: str, body: object = None, *, authorization: str | None = None
) -> tuple[int, bytes]:
    connection = http.client.HTTPSConnection(LOOPBACK, worker.running.port, timeout=60, context=unverified_tls())
    given = f"Bearer {worker.running.key.token}" if authorization is None else authorization
    sent = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
    try:
        connection.request(method, path, sent, {"Authorization": given} if given else {})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


@final
class CannedServer(http.server.ThreadingHTTPServer):
    answers: Mapping[str, tuple[int, bytes]] = {}
    said: Mapping[str, str] = {}


@final
class CannedAnswers(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._answered()

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._answered()

    def _answered(self) -> None:
        server = cast("CannedServer", self.server)
        status, body = server.answers[self.path]
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        for name, value in server.said.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def canned(
    key: WorkerKey, answers: Mapping[str, tuple[int, bytes]], said: Mapping[str, str] | None = None
) -> Generator[WorkerAddress]:
    server = CannedServer((LOOPBACK, 0), CannedAnswers)
    server.answers, server.said = answers, said or {}
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.minimum_version = ssl.TLSVersion.TLSv1_2
    tls.load_cert_chain(key.certificate, key.private_key)
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield WorkerAddress(LOOPBACK, cast("tuple[str, int]", server.server_address)[1], key.token, key.pin)
    finally:
        server.shutdown()
        server.server_close()
        joined(thread, 30)


def nobody_listens() -> int:
    with socket.socket() as taken:
        taken.bind((LOOPBACK, 0))
        return cast("tuple[str, int]", taken.getsockname())[1]
