import dataclasses
import http.client
import json
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from importlib.metadata import version
from types import MappingProxyType
from typing import cast

from sift.api.serving import ServeHome
from sift.interfaces.acts.acting import Acting
from sift.interfaces.server.app import ALIVE_SECONDS
from sift.interfaces.server.handled import HANDLED, Handled
from sift.interfaces.server.serving import HomeServer, held
from sift.settings import Settings

VERSION = version("sift-rag")


@contextmanager
def running(
    settings: Settings,
    handled: Mapping[str, Handled] = HANDLED,
    of: str = VERSION,
    alive_every: float = ALIVE_SECONDS,
) -> Generator[HomeServer]:
    with held(settings, ServeHome(port=0), of, handled, alive_every) as server:
        thread = threading.Thread(target=server.server.run, kwargs={"sockets": [server.socket]}, daemon=True)
        thread.start()
        waited = threading.Event()
        for _ in range(3000):
            if server.server.started:
                break
            waited.wait(0.01)
        try:
            yield server
        finally:
            server.server.should_exit = True
            thread.join(30)


type Line = dict[str, object]


def asked(
    server: HomeServer, name: str, body: object, *, authorization: str | None = None, host: str | None = None
) -> tuple[int, list[Line]]:
    connection = http.client.HTTPConnection("127.0.0.1", server.endpoint.port, timeout=60)
    given = f"Bearer {server.endpoint.token}" if authorization is None else authorization
    headers = {"Authorization": given} if given else {}
    if host is not None:
        headers["Host"] = host
    sent = body if isinstance(body, bytes) else json.dumps(body).encode()
    try:
        connection.request("POST", f"/api/{name}", sent, headers)
        response = connection.getresponse()
        status, read = response.status, response.read()
    finally:
        connection.close()
    if not read.startswith(b"{"):
        return status, [{"text": read.decode()}]
    return status, [cast("Line", json.loads(line)) for line in read.split(b"\n") if line]


def last(lines: list[Line], key: str) -> Line:
    return cast("Line", lines[-1][key])


def counted(asked: list[str]) -> Mapping[str, Handled]:
    def each(name: str, handled: Handled) -> Handled:
        def run(acting: Acting, data: object) -> object:
            asked.append(name)
            return handled.run(acting, data)

        return dataclasses.replace(handled, run=run)

    return MappingProxyType({name: each(name, handled) for name, handled in HANDLED.items()})
