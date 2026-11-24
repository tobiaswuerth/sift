import http.client
import signal
import sys
import threading
from pathlib import Path

import pytest

from sift.interfaces.server.endpoint import ENDPOINT, found
from sift.settings import Settings
from tests.surface import process
from tests.surface.commands import ran


def answers(home: Path) -> bool:
    endpoint = found(home)
    if endpoint is None:
        return False
    connection = http.client.HTTPConnection("127.0.0.1", endpoint.port, timeout=10)
    try:
        connection.request("POST", "/api/status", b"{}", {"Authorization": f"Bearer {endpoint.token}"})
        return connection.getresponse().status == http.client.OK
    except OSError:
        return False
    finally:
        connection.close()


def interrupted_once_answering(home: Path) -> None:
    waited = threading.Event()
    for _ in range(3000):
        if answers(home):
            signal.raise_signal(signal.SIGINT)
            return
        waited.wait(0.01)


def test_ctrl_c_stops_sift_serve_saying_so_and_leaves_nothing_behind(settings: Settings) -> None:
    threading.Thread(target=interrupted_once_answering, args=(settings.home,), daemon=True).start()
    code, out, err = ran(settings, "serve", "--port", "0")
    url = err.split("  ")[1]
    assert (code, out, err) == (0, "", f"serve  {url}  {settings.home}\n done  stopped serving {url}\n")
    assert (found(settings.home), (settings.home / ENDPOINT).exists()) == (None, False)
    said = next((settings.home / "logs").glob("serve-*.log")).read_text(encoding="utf-8")
    assert said.endswith(" INFO sift: ended with exit code 0\n")
    assert " INFO uvicorn.error: Started server process [" in said


@pytest.mark.skipif(sys.platform == "win32", reason="Windows gives a child process no Ctrl+C of its own")
def test_ctrl_c_stops_the_home_process_and_takes_its_endpoint_away(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with process.served(home) as running:
        assert (home / process.ENDPOINT).exists()
    assert (running.returncode, (home / process.ENDPOINT).exists(), found(home)) == (0, False, None)
