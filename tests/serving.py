import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from importlib.metadata import version

from sift.api.serving import ServeHome
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
